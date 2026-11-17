"""Heedle drafts: replaying a steering session without the instrument.

`docs/protocol.md` is the specification of `heedle-draft/1`. This file is the reference
verifier. It rebuilds every fell from the host's baseline and the recorded moves, and checks
each event as the specification lists. Each check carries the specification's label (H1,
E3, ...), and `tests/test_draft.py` has a tampered draft for every one.

A draft that breaks a rule of the specification and still passes here is a security bug.
"""
from __future__ import annotations

import base64
import hashlib
import re
from typing import List, Tuple

import numpy as np

from . import core
from .hosts import Host, host_from_descriptor

SCHEMA = "heedle-draft/1"
KINDS = ("pick", "snap", "unpick")
RTOL = 1e-6            # recorded distances, realized changes and exhaustive readings
RTOL_SEARCH = 1e-3     # the largest change a search found: its last steps can differ between machines
PIN_ATOL = 1e-6        # first-order change allowed at a pinned input
DECIMAL = re.compile(r"\d+(\.\d+)?")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _close(a: float, b: float, rtol: float) -> bool:
    return abs(float(a) - float(b)) <= rtol * max(1.0, abs(float(b)))


def replay(draft: dict, host: Host | None = None) -> List[str]:
    """Replay a draft and return a list of differences (empty when it checks out).

    A malformed draft is a difference, not an exception.
    """
    problems: List[str] = []
    try:
        _replay(draft, host, problems)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as err:
        problems.append(f"malformed draft: {type(err).__name__}: {err}")
    return problems


def _replay(draft: dict, host: Host | None, problems: List[str]) -> None:
    # H1: the schema is this one.
    if not isinstance(draft, dict) or draft.get("schema") != SCHEMA:
        problems.append(f"H1: unknown schema {draft.get('schema') if isinstance(draft, dict) else None!r}")
        return
    # H2: the host descriptor names a host.
    try:
        host = host or host_from_descriptor(draft["host"])
    except ValueError as err:
        problems.append(f"H2: {err}")
        return
    if draft["host"] != host.descriptor():
        # H3: and the baseline it states is the host's.
        problems.append("H3: the host descriptor is not this host's (kind, name or seed, baseline root)")
        return
    # H4: the limits are the host's. A draft does not choose its own.
    gate = draft["gate"]
    if gate != host.gate_header():
        problems.append(f"H4: the session's limits are not the host's ({host.gate_header()})")
        return
    sh = host.shape
    r_inv, cost, tau_text = host.r_invariant, float(gate["cost"]), gate["tau"]
    tau = float(tau_text)
    base = host.w0
    fit = host.encode(host.fit_inputs())
    # The fell after each event, so an unpick can return to an earlier one.
    lineage: List[Tuple[str, np.ndarray, float]] = [(draft["host"]["w0_root"], base.copy(), float(gate["budget"]))]
    prev = core.genesis_id(draft["host"], gate)
    for index, ev in enumerate(draft["events"]):
        tag = f"event {index + 1}"
        before = len(problems)
        fell_root, fell, budget = lineage[-1]
        # E1: events are numbered from 1.
        if not _is_int(ev.get("seq")) or ev["seq"] != index + 1:
            problems.append(f"{tag}: E1: seq is not {index + 1}")
        # E2: each event chains from the one before, and the first from the header.
        if ev.get("prev") != prev:
            problems.append(f"{tag}: E2: chain broken")
        # E3: an event's identifier is the hash of its own committed fields.
        if core.event_id(ev) != ev.get("id"):
            problems.append(f"{tag}: E3: identifier does not match its contents")
        prev = ev.get("id")
        # E4: an event starts from the fell.
        if ev.get("parent_root") != fell_root:
            problems.append(f"{tag}: E4: parent is not the fell")
        # E5: kind, gesture, limit and pins are well formed.
        kind = ev.get("kind")
        if kind not in KINDS:
            problems.append(f"{tag}: E5: unknown kind {kind!r}")
            return
        if ev.get("gesture") not in core.GESTURES:
            problems.append(f"{tag}: E5: unknown gesture {ev.get('gesture')!r}")
        if ev.get("tau") != tau_text or not DECIMAL.fullmatch(tau_text) or not tau > 0:
            problems.append(f"{tag}: E5: the event's limit is not the session's")
        pins = ev.get("pins")
        try:
            pins_ok = isinstance(pins, list) and all(_is_int(p) for p in pins) and host.normalise_pins(pins) == pins
        except ValueError:
            pins_ok = False
        if not pins_ok:
            problems.append(f"{tag}: E5: pins are not sorted, distinct integers of the host's domain")
            return
        if len(problems) > before:
            return                      # what follows would be read on a wrong footing

        if kind == "unpick":
            # U1: an unpick returns to a root that was the fell before.
            roots = [r for r, _, _ in lineage]
            if ev.get("root") not in roots:
                problems.append(f"{tag}: U1: unpick to a root that was never the fell")
                return
            at = len(roots) - 1 - roots[::-1].index(ev["root"])
            # The weights return. The budget already spent stays spent.
            lineage.append((ev["root"], lineage[at][1].copy(), budget))
            continue

        # M1: the move is one float64 per weight, and its hash is the recorded one.
        raw = base64.b64decode(ev["move_b64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != ev.get("move_sha256"):
            problems.append(f"{tag}: M1: move bytes do not match their hash")
        d = core.move_from_b64(ev["move_b64"], sh.P)
        cand = fell + d
        # G1: the gate, run here, gives the recorded verdict. A refused move is never an event.
        ok, dist, remaining, reason = core.verify_weight_update_safety(base, cand, r_inv, budget, cost)
        rec = ev["gate"]
        if rec.get("accepted") is not ok or rec.get("reason") != reason:
            problems.append(f"{tag}: G1: gate verdict differs ({reason}, recorded {rec.get('reason')})")
        if not ok:
            problems.append(f"{tag}: G1: a move the gate refuses was recorded")
            return
        # G2: the recorded distance and remaining budget are the gate's.
        if not _close(rec["distance"], dist, RTOL):
            problems.append(f"{tag}: G2: recorded distance differs")
        if not _close(rec["remaining_budget"], remaining, 1e-9):
            problems.append(f"{tag}: G2: recorded remaining budget differs")
        # R1: the candidate's weight root is the recorded root.
        root = host.root(cand)
        if root != ev.get("root"):
            problems.append(f"{tag}: R1: weight root differs")
        # P1: a move made with heddles raised leaves them alone to first order.
        if pins:
            y = core.jvp(sh, fell, host.encode(host.pin_inputs(pins)), d)
            if float(np.max(np.sqrt(np.mean(y * y, axis=1)))) > PIN_ATOL:
                problems.append(f"{tag}: P1: the move is not in the shed of its pins")
        # T1: the gesture is deliberate enough for the tension found here.
        tension = max(dist / r_inv, core.predicted_worst(sh, fell, fit, d) / tau)
        near = any(abs(tension - t) < 1e-6 for t in (0.3, 0.7))
        if not near and core.GESTURES.index(ev["gesture"]) < core.ZONES.index(core.zone_of(tension)):
            problems.append(f"{tag}: T1: gesture {ev['gesture']!r} is too light for tension {tension:.3f}")
        # C1: the claim is the one these roots, this limit and these pins make.
        cl = ev["claim"]
        cid = core.claim_id(host.kind, ev["parent_root"], root, tau_text, pins)
        if cid != cl.get("id"):
            problems.append(f"{tag}: C1: claim identifier differs")
        # C2: the fresh probes are the host's number of them, and the change on them is as recorded.
        if cl.get("probes") != host.probes:
            problems.append(f"{tag}: C2: not the host's number of fresh probes")
        probes = host.encode(host.fresh_inputs(core.probe_seed(root, cid), host.probes))
        if not _close(cl["realized"], core.behaviour_change(sh, fell, cand, probes), RTOL):
            problems.append(f"{tag}: C2: realized change differs")
        # C3: the draft says truly what the dispute read.
        if cl.get("exhaustive") is not host.exhaustive or cl.get("rows") != host.dispute_rows:
            problems.append(f"{tag}: C3: the draft misstates what the dispute read")
        # C4: the recorded witness is an input of the host, and changes by the recorded amount.
        seen = host.witness_change(fell, cand, cl["witness"])
        if not _close(cl["worst"], seen, RTOL):
            problems.append(f"{tag}: C4: the witness does not change by the recorded amount ({seen:.9g})")
        if kind == "snap":
            # S1: one input refutes the claim. The fell does not move; the budget is spent.
            if not seen > tau:
                problems.append(f"{tag}: S1: the witness does not refute the claim ({seen:.9g})")
            if cl.get("refuted") is not True:
                problems.append(f"{tag}: S1: a snap must carry a refuted claim")
            lineage.append((fell_root, fell, remaining))
        else:
            # K1: no input refutes the claim. The verifier runs the host's dispute itself.
            worst, _, _, exhaustive = host.dispute(fell, cand)
            if worst > tau or cl.get("refuted") is not False:
                problems.append(f"{tag}: K1: a refuted claim was admitted as a pick ({worst:.9g})")
            # K2: and finds the largest change the draft recorded.
            if not _close(cl["worst"], worst, RTOL if exhaustive else RTOL_SEARCH):
                problems.append(f"{tag}: K2: dispute reading differs ({worst:.9g})")
            lineage.append((root, cand, remaining))
