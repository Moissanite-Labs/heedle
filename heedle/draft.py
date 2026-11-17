"""Heedle drafts: replaying a steering session without the instrument.

`docs/protocol.md` is the specification of `heedle-draft/1`. This file is the reference
verifier. It rebuilds every fell from the baseline and the recorded moves, and checks each
event as the specification lists.
"""
from __future__ import annotations

import base64
import hashlib
from typing import List, Tuple

import numpy as np

from . import core
from .hosts import Host, host_from_descriptor

SCHEMA = "heedle-draft/1"


def replay(draft: dict, rtol: float = 1e-6, host: Host | None = None) -> List[str]:
    """Replay a draft and return a list of differences (empty when it checks out).

    Checked exactly: the baseline root, each move's hash, every weight root, the chain of
    event identifiers, the gate's verdict, and the witness of every snap (one input, both
    forward passes). Checked within a tolerance: recorded distances and readings, and that
    each gesture was deliberate enough for the tension found here.

    Not checked, because it cannot be: that a person made the gesture.
    """
    if draft.get("schema") != SCHEMA:
        return [f"unknown schema {draft.get('schema')!r}"]
    problems: List[str] = []
    host = host or host_from_descriptor(draft["host"])
    sh = host.shape
    if draft["host"].get("kind") != host.kind:
        problems.append("the draft names a different kind of host")
    if host.root(host.w0) != draft["host"]["w0_root"]:
        problems.append("baseline root does not match the host")
    gate = draft["gate"]
    r_inv, cost, budget = float(gate["r_invariant"]), float(gate["cost"]), float(gate["budget"])
    base = host.w0
    fit = host.encode(host.fit_inputs())
    # The fell after each event, so an unpick can return to an earlier one.
    lineage: List[Tuple[str, np.ndarray, float]] = [(draft["host"]["w0_root"], base.copy(), budget)]
    prev = "genesis"
    for ev in draft["events"]:
        tag = f"event {ev.get('seq')}"
        if ev.get("prev") != prev:
            problems.append(f"{tag}: chain broken")
        if core.event_id(ev) != ev.get("id"):
            problems.append(f"{tag}: identifier does not match its contents")
        prev = ev.get("id")
        fell_root, fell, budget = lineage[-1]
        if ev.get("parent_root") != fell_root:
            problems.append(f"{tag}: parent is not the fell")
        kind = ev.get("kind")
        if kind in ("pick", "snap"):
            raw = base64.b64decode(ev["move_b64"])
            if hashlib.sha256(raw).hexdigest() != ev["move_sha256"]:
                problems.append(f"{tag}: move bytes do not match their hash")
            d = core.move_from_b64(ev["move_b64"], sh.P)
            cand = fell + d
            ok, dist, remaining, reason = core.verify_weight_update_safety(base, cand, r_inv, budget, cost)
            rec = ev["gate"]
            if bool(rec["accepted"]) != ok or rec["reason"] != reason:
                problems.append(f"{tag}: gate verdict differs ({reason}, recorded {rec['reason']})")
            if not ok:
                problems.append(f"{tag}: a move the gate refuses was recorded")
                continue
            if abs(float(rec["distance"]) - dist) > rtol * max(1.0, dist):
                problems.append(f"{tag}: recorded distance differs")
            root = host.root(cand)
            if root != ev["root"]:
                problems.append(f"{tag}: weight root differs")
            tau = float(ev["tau"])
            tension = max(dist / r_inv, core.predicted_worst(sh, fell, fit, d) / tau)
            near = any(abs(tension - t) < 1e-6 for t in (0.3, 0.7))
            if not near and core.GESTURES.index(ev["gesture"]) < core.ZONES.index(core.zone_of(tension)):
                problems.append(f"{tag}: gesture {ev['gesture']!r} is too light for tension {tension:.3f}")
            cl = ev["claim"]
            cid = core.claim_id(host.kind, ev["parent_root"], root, ev["tau"], ev["pins"])
            if cid != cl["id"]:
                problems.append(f"{tag}: claim identifier differs")
            probes = host.encode(host.fresh_inputs(core.probe_seed(root, cid), int(cl["probes"])))
            realized = core.behaviour_change(sh, fell, cand, probes)
            if abs(realized - float(cl["realized"])) > rtol * max(1.0, realized):
                problems.append(f"{tag}: realized change differs ({realized:.9g})")
            if kind == "snap":
                try:
                    seen = host.witness_change(fell, cand, cl["witness"])
                except ValueError as err:
                    seen = float("nan")
                    problems.append(f"{tag}: {err}")
                if not seen > tau:
                    problems.append(f"{tag}: the witness does not refute the claim ({seen:.9g})")
                if not cl["refuted"]:
                    problems.append(f"{tag}: a snap must carry a refuted claim")
                lineage.append((fell_root, fell, remaining))
            else:
                worst, _, _, exhaustive = host.dispute(fell, cand)
                if abs(worst - float(cl["worst"])) > 1e-3 * max(1.0, worst):
                    problems.append(f"{tag}: dispute reading differs ({worst:.9g})")
                if worst > tau or cl["refuted"]:
                    problems.append(f"{tag}: a refuted claim was admitted as a pick")
                if bool(cl.get("exhaustive")) != exhaustive:
                    problems.append(f"{tag}: the draft misstates whether the dispute was exhaustive")
                lineage.append((root, cand, remaining))
        elif kind == "unpick":
            roots = [r for r, _, _ in lineage]
            if ev["root"] not in roots:
                problems.append(f"{tag}: unpick to a root that was never the fell")
                continue
            at = len(roots) - 1 - roots[::-1].index(ev["root"])
            # Unpicking returns the weights; the budget already spent stays spent.
            lineage.append((ev["root"], lineage[at][1].copy(), budget))
        else:
            problems.append(f"{tag}: unknown kind {kind!r}")
    return problems
