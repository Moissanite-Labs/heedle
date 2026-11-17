"""Tests for the draft verifier: recorded sessions replay, and tampering is caught.

    python3 -m unittest discover -s tests -p 'test_*.py'

The drafts in drafts/ were made by the JavaScript loom (tools/make_sample_drafts.mjs). Here
the Python reference rebuilds every fell from them.
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from heedle import core, draft  # noqa: E402
from heedle.hosts import host_from_descriptor  # noqa: E402


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drafts = {n: json.loads((REPO / "drafts" / f"sample-{n}.json").read_text()) for n in ("cloth", "text")}
        cls.hosts = {n: host_from_descriptor(d["host"]) for n, d in cls.drafts.items()}

    def replay(self, name, doc):
        return draft.replay(doc, host=self.hosts[name])

    @staticmethod
    def rechain(doc, start):
        """Recompute identifiers from `start` on, as someone rewriting the record would."""
        for i in range(start, len(doc["events"])):
            ev = doc["events"][i]
            if i > 0:
                ev["prev"] = doc["events"][i - 1]["id"]
            ev["id"] = core.event_id(ev)

    def test_the_recorded_sessions_replay(self):
        for name, doc in self.drafts.items():
            with self.subTest(host=name):
                self.assertEqual(self.replay(name, doc), [])
                kinds = [e["kind"] for e in doc["events"]]
                self.assertIn("snap", kinds)
                self.assertIn("unpick", kinds)

    def test_text_picks_were_checked_on_the_whole_domain(self):
        for ev in self.drafts["text"]["events"]:
            if ev["kind"] in ("pick", "snap"):
                self.assertTrue(ev["claim"]["exhaustive"])
                self.assertEqual(ev["claim"]["rows"], 28 ** 3)
        for ev in self.drafts["cloth"]["events"]:
            if ev["kind"] in ("pick", "snap"):
                self.assertFalse(ev["claim"]["exhaustive"])

    def test_a_changed_move_is_caught(self):
        for name in self.drafts:
            with self.subTest(host=name):
                doc = copy.deepcopy(self.drafts[name])
                ev = doc["events"][1]
                move = core.move_from_b64(ev["move_b64"], self.hosts[name].shape.P).copy()
                move[0] += 1e-9
                ev["move_b64"] = core.move_to_b64(move)
                problems = self.replay(name, doc)
                self.assertTrue(any("move bytes" in p for p in problems))
                self.assertTrue(any("weight root" in p for p in problems))

    def test_a_relabelled_or_reordered_event_is_caught(self):
        doc = copy.deepcopy(self.drafts["cloth"])
        doc["events"][2]["gesture"] = "knot"
        self.assertTrue(any("identifier" in p for p in self.replay("cloth", doc)))
        doc = copy.deepcopy(self.drafts["cloth"])
        doc["events"][1], doc["events"][2] = doc["events"][2], doc["events"][1]
        self.assertTrue(any("chain broken" in p for p in self.replay("cloth", doc)))

    def test_a_snap_needs_a_witness_that_refutes(self):
        # Point the recorded witness at an input that barely changed and the snap no longer
        # stands, even with every identifier consistent.
        for name in self.drafts:
            with self.subTest(host=name):
                doc = copy.deepcopy(self.drafts[name])
                host = self.hosts[name]
                at = next(i for i, e in enumerate(doc["events"]) if e["kind"] == "snap")
                snap = doc["events"][at]
                fell = host.w0.copy()
                for e in doc["events"][:at]:
                    if e["kind"] == "pick":
                        fell = fell + core.move_from_b64(e["move_b64"], host.shape.P)
                cand = fell + core.move_from_b64(snap["move_b64"], host.shape.P)
                quiet = [0.5, 0.5] if name == "cloth" else [int(np.argmin(core.change(
                    host.shape, fell, cand, host.encode(np.arange(host.pin_count)))))]
                if host.witness_change(fell, cand, quiet) > float(snap["tau"]):
                    self.skipTest("the chosen input happens to refute as well")
                snap["claim"]["witness"] = quiet
                self.assertTrue(any("does not refute" in p for p in self.replay(name, doc)))

    def test_a_light_gesture_on_a_tense_pick_is_caught(self):
        # Even with the chain recomputed, the gesture must match the tension the replay finds.
        for name in self.drafts:
            with self.subTest(host=name):
                doc = copy.deepcopy(self.drafts[name])
                at = next(i for i, e in enumerate(doc["events"]) if e["kind"] == "pick" and e["reading"]["zone"] != "auto")
                doc["events"][at]["gesture"] = "release"
                self.rechain(doc, at)
                problems = self.replay(name, doc)
                self.assertTrue(any("too light" in p for p in problems))
                self.assertFalse(any("chain" in p or "identifier" in p for p in problems))

    def test_an_admitted_pick_cannot_hide_a_refuted_claim(self):
        # Relabel the snap as a pick, with a consistent chain: the verifier's own dispute finds the input.
        for name in self.drafts:
            with self.subTest(host=name):
                doc = copy.deepcopy(self.drafts[name])
                at = next(i for i, e in enumerate(doc["events"]) if e["kind"] == "snap")
                doc["events"][at]["kind"] = "pick"
                doc["events"][at]["claim"]["refuted"] = False
                self.rechain(doc, at)
                self.assertTrue(any("refuted claim was admitted" in p for p in self.replay(name, doc)))

    def test_a_draft_names_its_host(self):
        doc = copy.deepcopy(self.drafts["cloth"])
        doc["host"]["seed"] = 8
        self.assertTrue(any("baseline root" in p for p in draft.replay(doc)))
        self.assertEqual(draft.replay({"schema": "heedle-draft/0"}), ["unknown schema 'heedle-draft/0'"])


if __name__ == "__main__":
    unittest.main()
