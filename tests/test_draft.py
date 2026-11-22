"""Tests for the draft verifier: recorded sessions replay, and every rule of the
specification has a tampered draft that breaks it and is caught.

    python3 -m unittest discover -s tests -p 'test_*.py'

The drafts in drafts/ were made by the JavaScript loom (tools/make_sample_drafts.mjs). Here
the Python reference rebuilds every fell from them.

The tampering is done as a forger would do it: after a change, the identifiers and the chain
are recomputed, so a tampered draft is caught by the rule it breaks and not by a stale hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from heedle import core, draft  # noqa: E402
from heedle.hosts import host_from_descriptor  # noqa: E402

DRAFTS = {n: json.loads((REPO / "drafts" / f"sample-{n}.json").read_text()) for n in ("cloth", "text")}
HOSTS = {n: host_from_descriptor(d["host"]) for n, d in DRAFTS.items()}


def rechain(doc, start=0):
    """Recompute identifiers from `start` on, as someone rewriting the record would."""
    for i in range(start, len(doc["events"])):
        ev = doc["events"][i]
        ev["prev"] = doc["events"][i - 1]["id"] if i else core.genesis_id(doc["host"], doc["gate"])
        ev["id"] = core.event_id(ev)


def fell_before(name, doc, at):
    """The weights at the fell just before event `at`, by replaying the picks so far."""
    host = HOSTS[name]
    lineage = [(doc["host"]["w0_root"], host.w0.copy())]
    for e in doc["events"][:at]:
        if e["kind"] == "pick":
            lineage.append((e["root"], lineage[-1][1] + core.move_from_b64(e["move_b64"], host.shape.P)))
        elif e["kind"] == "unpick":
            lineage.append(next(l for l in lineage if l[0] == e["root"]))
    return lineage[-1][1]


def set_move(name, doc, at, move):
    """Replace an event's move and everything a forger can recompute from it."""
    host, ev = HOSTS[name], doc["events"][at]
    fell = fell_before(name, doc, at)
    cand = fell + move
    ev["move_b64"] = core.move_to_b64(move)
    ev["move_sha256"] = hashlib.sha256(np.asarray(move, dtype="<f8").tobytes()).hexdigest()
    ev["root"] = host.root(cand)
    reclaim(name, doc, at)


def reclaim(name, doc, at):
    """Recompute an event's claim identifier and its reading on the fresh probes."""
    host, ev = HOSTS[name], doc["events"][at]
    fell = fell_before(name, doc, at)
    cand = fell + core.move_from_b64(ev["move_b64"], host.shape.P)
    ev["claim"]["id"] = core.claim_id(host.kind, ev["parent_root"], ev["root"], ev["tau"], ev["pins"])
    probes = host.encode(host.fresh_inputs(core.probe_seed(ev["root"], ev["claim"]["id"]), host.probes))
    ev["claim"]["realized"] = core.behaviour_change(host.shape, fell, cand, probes)


def first(doc, kind, **where):
    return next(i for i, e in enumerate(doc["events"]) if e["kind"] == kind and all(e.get(k) == v for k, v in where.items()))


def quiet_witness(name, doc, at):
    """An input that barely changes at event `at`, with the change it does make."""
    host, ev = HOSTS[name], doc["events"][at]
    fell = fell_before(name, doc, at)
    cand = fell + core.move_from_b64(ev["move_b64"], host.shape.P)
    if name == "cloth":
        from heedle.hosts import grid
        pts = grid(40)
        ch = core.change(host.shape, fell, cand, host.encode(pts))
        w = [float(v) for v in pts[int(np.argmin(ch))]]
    else:
        ch = core.change(host.shape, fell, cand, host.encode(np.arange(host.pin_count)))
        w = [int(np.argmin(ch))]
    return w, host.witness_change(fell, cand, w)


# Each case: the rule it breaks, a short name, the draft it starts from, and what the forger does.
# A case returns True when identifiers should be left stale (the tampering is not re-hashed).
def t_schema(d): d["schema"] = "heedle-draft/0"
def t_host_seed(d): d["host"]["seed"] = 8
def t_host_extra(d): d["host"]["note"] = "x"
def t_gate_radius(d): d["gate"]["r_invariant"] = "1000000000"
def t_gate_all(d): d["gate"] = {"r_invariant": "1000000000", "budget": 1000000, "cost": 1, "tau": "0.0001"}
def t_gate_no_tau(d): del d["gate"]["tau"]
def t_gate_nan(d): d["gate"]["tau"] = "nan"
def t_gate_bool_cost(d):
    d["gate"]["cost"] = True
    return True
def t_unpick_with_claim(d):
    d["events"][first(d, "unpick")]["claim"] = {"id": "0" * 64}
def t_seq(d):
    for e in d["events"]: e["seq"] = 7
def t_swap(d):
    d["events"][1], d["events"][2] = d["events"][2], d["events"][1]
    return True
def t_stale_id(d):
    d["events"][2]["gesture"] = "knot"
    return True
def t_parent(d): d["events"][1]["parent_root"] = "0" * 64
def t_kind(d): d["events"][1]["kind"] = "weave"
def t_gesture(d): d["events"][first(d, "unpick")]["gesture"] = "shrug"
def t_event_tau(d):
    at = first(d, "pick")
    d["events"][at]["tau"] = "5"
    reclaim("cloth", d, at)
def t_pins_unsorted(d):
    d["events"][0]["pins"] = [4095, 100, 4095]
    reclaim("cloth", d, 0)
def t_pins_outside(d):
    d["events"][0]["pins"] = [-3]
    reclaim("cloth", d, 0)
def t_unpick_nowhere(d): d["events"][first(d, "unpick")]["root"] = "f" * 64
def t_move_bytes(d):
    ev = d["events"][1]
    move = core.move_from_b64(ev["move_b64"], HOSTS["cloth"].shape.P).copy()
    move[0] += 1e-9
    ev["move_b64"] = core.move_to_b64(move)
    return True
def t_gate_verdict(d): d["events"][0]["gate"]["accepted"] = False
def t_refused_move(d):
    rng = np.random.default_rng(0)
    move = rng.standard_normal(HOSTS["cloth"].shape.P)
    set_move("cloth", d, 0, 50.0 * move / np.linalg.norm(move))
    del d["events"][1:]
def t_distance(d): d["events"][0]["gate"]["distance"] *= 1.01
def t_remaining(d): d["events"][1]["gate"]["remaining_budget"] = 24
def t_root(d): d["events"][0]["root"] = "0" * 64
def t_pins_not_held(d):
    d["events"][0]["pins"] = [100]          # says a heddle was raised; the move was made without it
    reclaim("cloth", d, 0)
def t_light_gesture(d):
    at = next(i for i, e in enumerate(d["events"]) if e["kind"] == "pick" and e["reading"]["zone"] != "auto")
    d["events"][at]["gesture"] = "release"
def t_claim_id(d): d["events"][0]["claim"]["id"] = "0" * 64
def t_probe_count(d): d["events"][0]["claim"]["probes"] = 1
def t_realized(d): d["events"][0]["claim"]["realized"] *= 1.1
def t_exhaustive(d): d["events"][0]["claim"]["exhaustive"] = not d["events"][0]["claim"]["exhaustive"]
def t_rows(d): d["events"][0]["claim"]["rows"] = 1
def t_worst(d): d["events"][0]["claim"]["worst"] *= 0.5
def t_snap_quiet(name):
    def go(d):
        at = first(d, "snap")
        d["events"][at]["claim"]["witness"], d["events"][at]["claim"]["worst"] = quiet_witness(name, d, at)
    return go
def t_snap_unrefuted(d): d["events"][first(d, "snap")]["claim"]["refuted"] = False
def t_snap_as_pick(d):
    at = first(d, "snap")
    d["events"][at]["kind"] = "pick"
    d["events"][at]["claim"]["refuted"] = False
def t_pick_quiet(name):
    def go(d):
        d["events"][0]["claim"]["witness"], d["events"][0]["claim"]["worst"] = quiet_witness(name, d, 0)
    return go
def t_no_claim(d): del d["events"][0]["claim"]
def t_short_move(d): d["events"][0]["move_b64"] = d["events"][0]["move_b64"][:16]
def t_bad_witness(d): d["events"][0]["claim"]["witness"] = [0.5]
def t_host_path(d): d["host"]["name"] = "../hosts/text-v1"
def t_host_seed_fraction(d): d["host"]["seed"] = 7.9
def t_host_kind(d): d["host"]["kind"] = "cloth/2"
def t_seq_fraction(d): d["events"][0]["seq"] = 1.5

CASES = [
    ("H1", "another schema", "cloth", t_schema),
    ("H2", "a host name that is a path", "text", t_host_path),
    ("H2", "a seed that is not an integer", "cloth", t_host_seed_fraction),
    ("H2", "an unknown kind of host", "cloth", t_host_kind),
    ("H3", "another seed", "cloth", t_host_seed),
    ("H3", "an extra field in the host", "cloth", t_host_extra),
    ("H4", "a wider gate in the header", "cloth", t_gate_radius),
    ("H4", "every limit changed in the header", "cloth", t_gate_all),
    ("H4", "no limit in the header", "cloth", t_gate_no_tau),
    ("H4", "a limit that is not a number", "cloth", t_gate_nan),
    ("H4", "a boolean cost equal to 1", "cloth", t_gate_bool_cost),
    ("E1", "events renumbered", "cloth", t_seq),
    ("E2", "two events swapped", "cloth", t_swap),
    ("E3", "a field changed under its identifier", "cloth", t_stale_id),
    ("E4", "a parent that is not the fell", "cloth", t_parent),
    ("E5: unknown kind", "an unknown kind", "cloth", t_kind),
    ("E5: unknown gesture", "an unknown gesture on an unpick", "cloth", t_gesture),
    ("E5: the event's limit", "an event with its own, looser limit", "cloth", t_event_tau),
    ("E5: pins", "pins unsorted and repeated", "cloth", t_pins_unsorted),
    ("E5: pins", "a pin outside the domain", "cloth", t_pins_outside),
    ("U1", "an unpick to a root that never was", "cloth", t_unpick_nowhere),
    ("U1", "an unpick carrying a claim", "cloth", t_unpick_with_claim),
    ("M1", "a move changed under its hash", "cloth", t_move_bytes),
    ("G1: gate verdict", "a recorded refusal the gate does not give", "cloth", t_gate_verdict),
    ("G1: a move the gate refuses", "a move far outside the gate", "cloth", t_refused_move),
    ("G2: recorded distance", "a wrong distance", "cloth", t_distance),
    ("G2: recorded remaining budget", "a budget that was not spent", "cloth", t_remaining),
    ("R1", "a wrong root", "cloth", t_root),
    ("P1", "a heddle claimed but not held", "cloth", t_pins_not_held),
    ("T1", "a light gesture on a tense pick", "cloth", t_light_gesture),
    ("T1", "a light gesture on a tense pick", "text", t_light_gesture),
    ("C1", "a wrong claim identifier", "cloth", t_claim_id),
    ("C2: not the host's number", "one fresh probe", "cloth", t_probe_count),
    ("C2: realized", "a wrong reading on the fresh probes", "cloth", t_realized),
    ("C3", "a search called exhaustive", "cloth", t_exhaustive),
    ("C3", "an exhaustive reading called a search", "text", t_exhaustive),
    ("C3", "a dispute said to read one input", "text", t_rows),
    ("C4", "a witness that does not change by the recorded amount", "cloth", t_worst),
    ("S1: the witness does not refute", "a snap pointed at a quiet input", "cloth", t_snap_quiet("cloth")),
    ("S1: the witness does not refute", "a snap pointed at a quiet input", "text", t_snap_quiet("text")),
    ("S1: a snap must", "a snap whose claim is not refuted", "cloth", t_snap_unrefuted),
    ("K1", "a snap relabelled as a pick", "cloth", t_snap_as_pick),
    ("K1", "a snap relabelled as a pick", "text", t_snap_as_pick),
    ("K2", "a pick that understates its largest change", "cloth", t_pick_quiet("cloth")),
    ("K2", "a pick that understates its largest change", "text", t_pick_quiet("text")),
    ("malformed", "an event with no claim", "cloth", t_no_claim),
    ("malformed", "a move cut short", "cloth", t_short_move),
    ("malformed", "a witness with one number", "cloth", t_bad_witness),
    ("malformed", "a fraction in a hashed field", "cloth", t_seq_fraction),
]


class Replay(unittest.TestCase):
    def test_the_recorded_sessions_replay(self):
        for name, doc in DRAFTS.items():
            with self.subTest(host=name):
                self.assertEqual(draft.replay(doc, host=HOSTS[name]), [])
                kinds = [e["kind"] for e in doc["events"]]
                self.assertIn("snap", kinds)
                self.assertIn("unpick", kinds)
                self.assertTrue(any(e["pins"] for e in doc["events"]))

    def test_text_picks_were_checked_on_the_whole_domain(self):
        for ev in DRAFTS["text"]["events"]:
            if ev["kind"] != "unpick":
                self.assertTrue(ev["claim"]["exhaustive"])
                self.assertEqual(ev["claim"]["rows"], 28 ** 3)
        for ev in DRAFTS["cloth"]["events"]:
            if ev["kind"] != "unpick":
                self.assertFalse(ev["claim"]["exhaustive"])

    def test_every_rule_has_a_tampered_draft_that_is_caught(self):
        for label, what, name, tamper in CASES:
            with self.subTest(rule=label, what=what, host=name):
                doc = copy.deepcopy(DRAFTS[name])
                stale = tamper(doc)
                if not stale and doc.get("schema") == draft.SCHEMA and "events" in doc:
                    try:
                        rechain(doc)
                    except (ValueError, KeyError, TypeError):
                        pass          # a draft that cannot be hashed stays as the forger left it
                problems = draft.replay(doc, host=None if label in ("H2", "H3") else HOSTS[name])
                self.assertTrue(any(label in p for p in problems), f"{what}: expected {label}, got {problems}")

    def test_every_rule_in_the_verifier_is_covered(self):
        # Each label the verifier can report must be the target of at least one case above.
        source = (REPO / "heedle" / "draft.py").read_text()
        reported = set(re.findall(r'problems\.append\(f?"(?:\{tag\}: )?([A-Z]\d)', source)) | {"malformed"}
        covered = {label.split(":")[0] for label, *_ in CASES}
        self.assertEqual(reported - covered, set())
        self.assertGreaterEqual(len(reported), 18)

    def test_forgeries_from_the_first_review_are_caught(self):
        # A: a refuted pick recorded as admitted by giving the event its own, looser limit.
        for name, loose in (("cloth", "5"), ("text", "1.8")):
            with self.subTest(forgery="A", host=name):
                doc = copy.deepcopy(DRAFTS[name])
                at = first(doc, "snap")
                ev = doc["events"][at]
                ev.update(kind="pick", tau=loose, gesture="knot")
                ev["claim"]["refuted"] = False
                reclaim(name, doc, at)
                del doc["events"][at + 1:]
                rechain(doc)
                self.assertTrue(any("E5: the event's limit" in p for p in draft.replay(doc, host=HOSTS[name])))
        # C: any weights at all, by widening the header and the event's limit together.
        doc = copy.deepcopy(DRAFTS["cloth"])
        doc["gate"]["r_invariant"], doc["gate"]["tau"] = "1000000000", "1000000"
        rng = np.random.default_rng(1)
        move = rng.standard_normal(HOSTS["cloth"].shape.P)
        doc["events"][0]["tau"] = "1000000"
        set_move("cloth", doc, 0, 50.0 * move / np.linalg.norm(move))
        del doc["events"][1:]
        rechain(doc)
        self.assertTrue(any("H4" in p for p in draft.replay(doc, host=HOSTS["cloth"])))
        # D: the header alone, with no hash touched.
        doc = copy.deepcopy(DRAFTS["cloth"])
        doc["gate"] = {"r_invariant": "1000000000", "budget": 1000000, "cost": 1, "tau": "0.0001"}
        self.assertTrue(any("H4" in p for p in draft.replay(doc, host=HOSTS["cloth"])))

    def test_the_header_is_in_the_chain(self):
        doc = DRAFTS["cloth"]
        self.assertEqual(doc["events"][0]["prev"], core.genesis_id(doc["host"], doc["gate"]))
        other = dict(doc["gate"], tau="0.6")
        self.assertNotEqual(core.genesis_id(doc["host"], other), doc["events"][0]["prev"])
        with self.assertRaises(ValueError):
            core.canonical_json({"tau": 0.5})

    def test_t1_boundary_and_strict_type_hardening(self):
        host = HOSTS["cloth"]
        sh = host.shape
        fit = host.encode(host.fit_inputs())
        th = host.threading(host.w0)
        base_dir = th.shaft_move(0, 1.0)
        unit_tension = max(
            float(np.linalg.norm(base_dir)) / host.r_invariant,
            core.predicted_worst(sh, host.w0, fit, base_dir) / host.tau_value,
        )

        def make_pick_at_tension(target_tension: float, gesture: str) -> dict:
            move = base_dir * (target_tension / unit_tension)
            cand = host.w0 + move
            ok, dist, remaining, reason = core.verify_weight_update_safety(
                host.w0, cand, host.r_invariant, float(host.budget), float(host.cost)
            )
            self.assertTrue(ok)
            w0_root = host.root(host.w0)
            root = host.root(cand)
            cid = core.claim_id(host.kind, w0_root, root, host.tau, [])
            probes = host.encode(host.fresh_inputs(core.probe_seed(root, cid), host.probes))
            realized = core.behaviour_change(sh, host.w0, cand, probes)
            worst, witness, rows, exhaustive = host.dispute(host.w0, cand)
            ev = {
                "seq": 1,
                "kind": "snap" if worst > host.tau_value else "pick",
                "parent_root": w0_root,
                "root": root,
                "move_b64": core.move_to_b64(move),
                "move_sha256": hashlib.sha256(np.asarray(move, dtype="<f8").tobytes()).hexdigest(),
                "pins": [],
                "gesture": gesture,
                "tau": host.tau,
                "gate": {"accepted": True, "distance": dist, "remaining_budget": remaining, "reason": reason},
                "claim": {
                    "id": cid, "probes": host.probes, "realized": realized,
                    "worst": worst, "witness": witness, "rows": rows,
                    "exhaustive": exhaustive, "refuted": worst > host.tau_value,
                },
            }
            doc = {"schema": draft.SCHEMA, "host": host.descriptor(), "gate": host.gate_header(), "events": [ev]}
            rechain(doc)
            return doc

        # Near 0.7 (knot boundary): "knot" and adjacent "hold" pass, but two-zone skip "release" fails T1.
        self.assertEqual(draft.replay(make_pick_at_tension(0.7, "knot"), host=host), [])
        self.assertEqual(draft.replay(make_pick_at_tension(0.7, "hold"), host=host), [])
        probs_rel_07 = draft.replay(make_pick_at_tension(0.7, "release"), host=host)
        self.assertTrue(any("T1" in p for p in probs_rel_07), probs_rel_07)
        # Just below 0.7 (shared zone, within 1e-6 of 0.7): "hold" passes, "release" fails T1.
        self.assertEqual(draft.replay(make_pick_at_tension(0.7 - 5e-7, "hold"), host=host), [])
        probs_rel_below07 = draft.replay(make_pick_at_tension(0.7 - 5e-7, "release"), host=host)
        self.assertTrue(any("T1" in p for p in probs_rel_below07), probs_rel_below07)
        # Away from 0.7 (strictly above 0.7 + 1e-6): "hold" fails T1.
        probs_hold_above07 = draft.replay(make_pick_at_tension(0.7 + 1e-5, "hold"), host=host)
        self.assertTrue(any("T1" in p for p in probs_hold_above07), probs_hold_above07)
        # Near 0.3 (shared boundary): adjacent "release" passes; away from 0.3 it fails T1.
        self.assertEqual(draft.replay(make_pick_at_tension(0.3, "release"), host=host), [])
        probs_rel_above03 = draft.replay(make_pick_at_tension(0.3 + 1e-5, "release"), host=host)
        self.assertTrue(any("T1" in p for p in probs_rel_above03), probs_rel_above03)

        # Strict numeric / exact-type guards reject booleans, strings, non-finite/overflowing ints, and floats for int fields.
        self.assertFalse(draft._close(True, 1.0, 1e-6))
        self.assertFalse(draft._close("1.0", 1.0, 1e-6))
        self.assertFalse(draft._close(float("nan"), 1.0, 1e-6))
        self.assertFalse(draft._close(10**400, 1.0, 1e-6))
        self.assertFalse(draft._exact_equal({"cost": True}, {"cost": 1}))
        from heedle.hosts import ClothHost
        host1 = ClothHost(1)
        bad_h3 = {
            "schema": draft.SCHEMA,
            "host": {"kind": "cloth/1", "seed": True, "w0_root": host1.root(host1.w0)},
            "gate": host1.gate_header(),
            "events": [],
        }
        self.assertTrue(any("H3" in p for p in draft.replay(bad_h3, host=host1)))
        bad_bool_metric = make_pick_at_tension(0.3, "hold")
        bad_bool_metric["events"][0]["gate"]["distance"] = True
        self.assertTrue(any("G2" in p for p in draft.replay(bad_bool_metric, host=host)))
        bad_float_probes = make_pick_at_tension(0.3, "hold")
        bad_float_probes["events"][0]["claim"]["probes"] = float(host.probes)
        self.assertTrue(any("C2" in p for p in draft.replay(bad_float_probes, host=host)))
        bad_float_rows = make_pick_at_tension(0.3, "hold")
        bad_float_rows["events"][0]["claim"]["rows"] = float(host.dispute_rows)
        self.assertTrue(any("C3" in p for p in draft.replay(bad_float_rows, host=host)))


if __name__ == "__main__":
    unittest.main()
