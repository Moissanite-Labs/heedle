"""Tests for Heedle's reference implementation, on both hosts.

    python3 -m unittest discover -s tests -p 'test_*.py'

Each test states what it checks. A property that holds by construction is marked as such:
it tests the code, not an idea. docs/theory.md names these tests where it cites them.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from heedle import core  # noqa: E402
from heedle.hosts import ALPHABET, ClothHost, TextHost, normalise_text  # noqa: E402

SPECIAL = {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}
decode = lambda x: SPECIAL[x] if isinstance(x, str) and x in SPECIAL else x  # noqa: E731


class Bench:
    """One host with what several tests need, computed once."""
    cache: dict = {}

    def __init__(self, host):
        self.host, self.sh, self.w = host, host.shape, host.w0
        self.fit_inputs = host.fit_inputs()
        self.fit = host.encode(self.fit_inputs)
        self.th = host.threading(self.w)
        self.patterns = host.patterns(self.w, self.fit_inputs)

    @classmethod
    def all(cls):
        if not cls.cache:
            text = TextHost()
            cls.cache = {"cloth": cls(ClothHost(7)), "text": cls(text)}
            cls.cache["cloth"].pins = [100, 2000, 3000, 3001]
            cls.cache["text"].pins = [text.context_of(" th"), text.context_of("the")]
        return cls.cache.items()


class Numbers(unittest.TestCase):
    def test_generator_matches_the_vectors(self):
        vec = json.loads((REPO / "vectors" / "cloth.json").read_text())
        draw = core.mulberry32(12345)
        self.assertEqual([draw() for _ in range(8)], vec["exact"]["mulberry32_12345"])


class HostMaths(unittest.TestCase):
    def test_three_ways_to_differentiate_agree(self):
        # J d by the forward rule, J' r by the backward rule, and the dense Jacobian.
        rng = np.random.default_rng(0)
        for name, b in Bench.all():
            with self.subTest(host=name):
                enc = b.host.encode(b.fit_inputs[:6])
                J = core.jacobian(b.sh, b.w, enc)
                d, R = rng.standard_normal(b.sh.P), rng.standard_normal((enc.n, b.sh.M))
                Rc = R - R.mean(axis=1, keepdims=True) if b.sh.centre else R
                self.assertLess(np.max(np.abs(core.jvp(b.sh, b.w, enc, d).ravel() - J @ d)), 1e-10)
                self.assertLess(np.max(np.abs(core.vjp(b.sh, b.w, enc, R) - J.T @ Rc.ravel())), 1e-10)

    def test_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        for name, b in Bench.all():
            with self.subTest(host=name):
                enc = b.host.encode(b.fit_inputs[:5])
                d = rng.standard_normal(b.sh.P)
                d /= np.linalg.norm(d)
                eps = 1e-6
                fd = (core.forward(b.sh, b.w + eps * d, enc) - core.forward(b.sh, b.w - eps * d, enc)) / (2 * eps)
                self.assertLess(np.max(np.abs(fd - core.jvp(b.sh, b.w, enc, d))), 1e-6)

    def test_centred_outputs_sum_to_zero(self):
        b = dict(Bench.all())["text"]
        self.assertTrue(b.sh.centre)
        self.assertLess(np.max(np.abs(core.forward(b.sh, b.w, b.fit).sum(axis=1))), 1e-12)
        d = np.random.default_rng(2).standard_normal(b.sh.P)
        self.assertLess(np.max(np.abs(core.jvp(b.sh, b.w, b.fit, d).sum(axis=1))), 1e-10)


class Shafts(unittest.TestCase):
    def test_shafts_are_orthonormal_and_ordered(self):
        for name, b in Bench.all():
            with self.subTest(host=name):
                V = b.th.V[:, :32]
                self.assertLess(np.max(np.abs(V.T @ V - np.eye(32))), 1e-9)
                self.assertTrue(np.all(np.diff(b.th.lam) <= 1e-9))

    def test_travel_is_measured_in_behaviour(self):
        # By construction: a move of travel t along shaft i has first-order change |t|.
        for name, b in Bench.all():
            with self.subTest(host=name):
                for i in (0, 5, 15):
                    for t in (0.2, -0.7):
                        self.assertAlmostEqual(b.th.predicted_change(b.th.shaft_move(i, t)), abs(t), places=9)

    def test_tie_up_delivers_its_share(self):
        for name, b in Bench.all():
            for K in (4, 16, 64):
                tie = core.TieUp(b.th, b.patterns, K)
                for treadle in b.host.treadle_names:
                    with self.subTest(host=name, K=K, treadle=treadle):
                        rho, t, c = tie.share[treadle], 0.37, b.patterns[treadle]
                        self.assertTrue(0.0 <= rho <= 1.0 + 1e-12)
                        dy = b.th.J @ tie.move({treadle: t})
                        # gain along the pattern is t * share; what is missing has RMS t * sqrt(1 - share)
                        self.assertAlmostEqual(core.gain_along(dy, c), t * rho, places=9)
                        self.assertAlmostEqual(math.sqrt(np.mean((t * c - dy) ** 2)), t * math.sqrt(max(0.0, 1 - rho)), places=8)

    def test_more_shafts_never_carry_less(self):
        for name, b in Bench.all():
            last = {n: 0.0 for n in b.host.treadle_names}
            for K in (2, 4, 8, 16, 32, 64, 128):
                tie = core.TieUp(b.th, b.patterns, K)
                for n in b.host.treadle_names:
                    with self.subTest(host=name, K=K, treadle=n):
                        self.assertGreaterEqual(tie.share[n], last[n] - 1e-12)
                        last[n] = tie.share[n]

    def test_a_treadling_is_a_weaving_draft(self):
        # To first order:  output change = threading x tie-up x treadling.
        for name, b in Bench.all():
            with self.subTest(host=name):
                K = 16
                tie = core.TieUp(b.th, b.patterns, K)
                travels = np.array([0.3, -0.2, 0.5, 0.0, 0.1])
                treadling = dict(zip(tie.names, travels))
                threading = b.th.U[:, :K] * math.sqrt(b.th.rows)
                drawdown = threading @ tie.matrix() @ travels
                self.assertLess(np.max(np.abs(b.th.J @ tie.move(treadling) - drawdown)), 1e-9)

    def test_worst_and_typical_directions_differ(self):
        # THEORY.md section 2 of the calibration record: the worst direction changes behaviour
        # sqrt(lam_max) per unit of weight length, a random one sqrt(trace(G)/P) on average.
        for name, b in Bench.all():
            with self.subTest(host=name):
                kappa = math.sqrt(b.th.lam[0] * b.sh.P / np.sum(b.th.lam))
                self.assertGreater(kappa, 4.0)


class Pins(unittest.TestCase):
    def test_projection_leaves_the_pins_alone_to_first_order(self):
        rng = np.random.default_rng(3)
        for name, b in Bench.all():
            with self.subTest(host=name):
                shed = b.host.shed(b.w, b.pins)
                p = shed.project(rng.standard_normal(b.sh.P))
                self.assertLess(np.max(np.abs(shed.J @ p)), 1e-9)
                self.assertLess(np.max(np.abs(shed.project(p) - p)), 1e-9)      # idempotent
                self.assertEqual(shed.open_dimensions, b.sh.P - np.linalg.matrix_rank(shed.J))

    def test_what_reaches_the_pins_is_second_order(self):
        # A move in the shed changes the pinned outputs by about c2 r^2: halving it quarters the leak.
        rng = np.random.default_rng(4)
        for name, b in Bench.all():
            with self.subTest(host=name):
                shed = b.host.shed(b.w, b.pins)
                enc = b.host.encode(b.host.pin_inputs(b.pins))
                d = shed.project(rng.standard_normal(b.sh.P))
                d /= np.linalg.norm(d)
                leak = [float(np.max(core.change(b.sh, b.w, b.w + r * d, enc))) for r in (0.02, 0.01)]
                self.assertTrue(3.5 < leak[0] / leak[1] < 4.5, leak)
                raw = rng.standard_normal(b.sh.P)
                raw /= np.linalg.norm(raw)
                self.assertGreater(float(np.max(core.change(b.sh, b.w, b.w + 0.01 * raw, enc))), 20 * leak[1])

    def test_threading_around_the_pins(self):
        # Shafts of G restricted to the shed leave the pins alone, and pins never add reach.
        for name, b in Bench.all():
            with self.subTest(host=name):
                shed = b.host.shed(b.w, b.pins)
                inside = b.host.threading(b.w, shed)
                self.assertLess(np.max(np.abs(shed.J @ inside.V[:, :16])), 1e-8)
                self.assertTrue(np.all(inside.lam[:16] <= b.th.lam[:16] + 1e-9))

    def test_no_pins_is_the_whole_space(self):
        for name, b in Bench.all():
            with self.subTest(host=name):
                shed = b.host.shed(b.w, [])
                d = np.arange(b.sh.P, dtype=np.float64)
                self.assertTrue(np.array_equal(shed.project(d), d))
                self.assertEqual(shed.open_dimensions, b.sh.P)


class Edges(unittest.TestCase):
    def test_gate_port_agrees_with_wefts_vectors(self):
        vec = json.loads((REPO / "vectors" / "weft-invariants.json").read_text())
        ran = 0
        for v in vec["weight_update_vectors"]:
            if v.get("locked_mask") is not None:
                continue   # the masked path is not ported: no instrument passes a mask
            got = core.verify_weight_update_safety([decode(x) for x in v["baseline"]], [decode(x) for x in v["candidate"]],
                                                   decode(v["r_invariant"]), decode(v["budget"]), decode(v["cost"]))
            self.assertEqual((got[0], got[3]), (v["expected_accepted"], v["expected_reason"]), v["name"])
            want = decode(v["expected_distance"])
            if math.isfinite(want):
                self.assertAlmostEqual(got[1], want, delta=1e-9 * max(1.0, abs(want)), msg=v["name"])
            else:
                self.assertEqual(got[1], want, v["name"])
            ran += 1
        self.assertGreaterEqual(ran, 10)

    def test_vendored_files_are_unchanged(self):
        for f in json.loads((REPO / "VENDORED.json").read_text())["files"]:
            self.assertEqual(hashlib.sha256((REPO / f["path"]).read_bytes()).hexdigest(), f["sha256"], f["path"])

    def test_the_proved_bound_is_never_exceeded(self):
        # Theorem 1 of the calibration theory, slope 1. A check of the port on moves of several
        # kinds and sizes. It is a proved bound: a failure here would be a bug, not a finding.
        rng = np.random.default_rng(5)
        for name, b in Bench.all():
            enc = b.host.encode(b.host.held_inputs()[:300])
            y0 = core.forward(b.sh, b.w, enc)
            for radius in (0.01, 0.1, 0.5, 2.0):
                bound = core.certified_bound(b.sh, b.w, enc, radius)
                for d in [rng.standard_normal(b.sh.P) for _ in range(3)] + [b.th.V[:, 0], -b.th.V[:, 1]]:
                    with self.subTest(host=name, radius=radius):
                        out = core.forward(b.sh, b.w + radius * d / np.linalg.norm(d), enc) - y0
                        self.assertTrue(np.all(np.linalg.norm(out, axis=1) <= bound))


def cbc_weight_root(tensors: dict) -> str:
    """The weight root of committed-behaviour-change v1, written out again from its design
    (weft-loom, experiments/committed-behaviour-change-2026-11-15/DESIGN.md and cbc.py), so
    that this test does not use the code it checks."""
    domain = b"WEFT-CBC-V1:"
    h = hashlib.sha256(domain + b"root:")
    for name in sorted(tensors):
        a = np.asarray(tensors[name])
        header = json.dumps({"name": name, "shape": list(a.shape), "dtype": a.dtype.str}, sort_keys=True, separators=(",", ":")).encode()
        leaf = hashlib.sha256(domain + b"leaf:" + header + b":" + np.ascontiguousarray(a).tobytes()).hexdigest()
        h.update(name.encode() + b":" + leaf.encode() + b";")
    return h.hexdigest()


class Commitments(unittest.TestCase):
    def test_weight_root_is_the_cbc_weight_root(self):
        for name, b in Bench.all():
            with self.subTest(host=name):
                self.assertEqual(b.host.root(b.w), cbc_weight_root(b.sh.tensors(b.w)))
                self.assertNotEqual(b.host.root(b.w + 1e-9), b.host.root(b.w))
        # The cloth's baseline for seed 7 is the one weft-loom's prototype test compared
        # against cbc.py itself.
        self.assertEqual(ClothHost(7).root(ClothHost(7).w0), "a554c6205b712f158e5d0c2e35aa53dabd7b1117ffd8b0a6e2b0232f6cbc5e75")

    def test_probes_follow_the_commitment(self):
        for name, b in Bench.all():
            with self.subTest(host=name):
                root = b.host.root(b.w)
                a = np.asarray(b.host.fresh_inputs(core.probe_seed(root, "c" * 64), 32))
                self.assertTrue(np.array_equal(a, b.host.fresh_inputs(core.probe_seed(root, "c" * 64), 32)))
                other = b.host.root(b.w + 1e-9)           # one weight changes them
                self.assertFalse(np.array_equal(a, b.host.fresh_inputs(core.probe_seed(other, "c" * 64), 32)))
                self.assertFalse(np.array_equal(a, b.host.fresh_inputs(core.probe_seed(root, "d" * 64), 32)))

    def test_cloth_dispute_is_at_least_the_lattice_reading(self):
        b = dict(Bench.all())["cloth"]
        w1 = b.w + b.th.shaft_move(0, 0.4)
        worst, witness, rows, exhaustive = b.host.dispute(b.w, w1)
        from heedle.hosts import grid
        lattice = float(np.max(core.change(b.sh, b.w, w1, b.host.encode(grid(b.host.DISPUTE_GRID)))))
        self.assertGreaterEqual(worst, lattice)
        self.assertFalse(exhaustive)
        self.assertAlmostEqual(b.host.witness_change(b.w, w1, witness), worst, places=12)   # one input is the whole evidence

    def test_text_dispute_reads_the_whole_domain(self):
        b = dict(Bench.all())["text"]
        w1 = b.w + b.th.shaft_move(0, 0.4)
        worst, witness, rows, exhaustive = b.host.dispute(b.w, w1)
        self.assertTrue(exhaustive)
        self.assertEqual(rows, 28 ** 3)
        everything = core.change(b.sh, b.w, w1, b.host.encode(np.arange(28 ** 3)))
        self.assertEqual(worst, float(everything.max()))        # nothing larger exists in the domain
        self.assertEqual(witness, [int(everything.argmax())])
        self.assertAlmostEqual(b.host.witness_change(b.w, w1, witness), worst, places=12)
        with self.assertRaises(ValueError):
            b.host.witness_inputs([28 ** 3])

    def test_a_move_can_hide_from_probes_it_knows(self):
        # A proposer who sees the probes can search the directions the probes cannot see.
        for name, b in Bench.all():
            with self.subTest(host=name):
                known, elsewhere = b.host.encode(b.host.known_inputs()), b.host.encode(b.host.elsewhere_inputs())
                d = core.hidden_move(b.sh, b.w, known, elsewhere, 0.3)
                self.assertAlmostEqual(float(np.linalg.norm(d)), 0.3, places=12)
                self.assertLess(np.max(np.abs(core.jvp(b.sh, b.w, known, d))), 1e-8)
                self.assertGreater(core.behaviour_change(b.sh, b.w, b.w + d, elsewhere),
                                   3 * core.behaviour_change(b.sh, b.w, b.w + d, known))


class TextHostFile(unittest.TestCase):
    def test_the_host_file_is_what_it_says(self):
        data = json.loads((REPO / "hosts" / "text-v1.json").read_text())
        host = TextHost()
        self.assertEqual(host.shape.P, 4700)
        self.assertEqual(host.root(host.w0), data["w0_root"])
        corpus = REPO / data["training"]["corpus"]["path"]
        self.assertEqual(hashlib.sha256(corpus.read_bytes()).hexdigest(), data["training"]["corpus"]["sha256"])
        text = " ".join(corpus.read_text().split("\n")).strip()
        self.assertEqual(len(text), data["training"]["corpus"]["characters"])
        self.assertTrue(set(text) <= set(ALPHABET))
        self.assertEqual(normalise_text(text), text)
        self.assertFalse(set(host.fit) & set(host.held))            # held-out contexts are held out
        self.assertLess(data["training"]["bits_per_character_after"], data["training"]["bits_per_character_unigram"])

    def test_contexts_round_trip(self):
        host = TextHost()
        for s in (" th", "the", "...", "zzz", "a b"):
            self.assertEqual(host.context_text(host.context_of(s)), s)
        self.assertEqual(len(host.weave(host.w0, "the", 40, 1)), 40)


if __name__ == "__main__":
    unittest.main()
