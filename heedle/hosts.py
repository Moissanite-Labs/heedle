"""Heedle hosts: what the inputs of a small model mean.

A host supplies everything `heedle/core.py` does not know: how inputs become features, which
inputs are the fit probes, what a pin is, how probes are drawn from a hash, how to look for
the one input that breaks a claim, and which named changes (treadles) a person can ask for.
`docs/hosts.md` states the contract and how to add a host.

Two hosts:

    cloth/1   a point of the unit square to a colour. 467 weights. Every input can be
              seen at once, so the picture is the model's whole behaviour. The dispute
              search reads a lattice and climbs: a sampled reading.
    text/1    three characters to the next one. 4,700 weights, trained on this repository's
              own documents. The domain is every one of 21,952 contexts, so the dispute
              search reads all of them: an exhaustive reading.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core
from .core import Enc, Shape

REPO = Path(__file__).resolve().parents[1]


class Host:
    """The contract. Subclasses fill in the inputs; everything here is shared."""

    kind: str
    shape: Shape
    w0: np.ndarray
    r_invariant: float
    tau: str
    budget: int = 24
    cost: int = 1
    probes: int = 64                 # fresh probes per pick
    travel: float = 0.6              # how far the instrument's knobs run, in units of behaviour
    treadle_names: List[str]

    # --- what a subclass provides ---
    def encode(self, inputs) -> Enc: raise NotImplementedError
    def fit_inputs(self): raise NotImplementedError
    def held_inputs(self): raise NotImplementedError
    def known_inputs(self): raise NotImplementedError
    def elsewhere_inputs(self): raise NotImplementedError
    def pin_inputs(self, pins: Sequence[int]): raise NotImplementedError
    def fresh_inputs(self, seed: bytes, n: int): raise NotImplementedError
    def treadle_pattern(self, name: str, inputs, y: np.ndarray) -> np.ndarray: raise NotImplementedError
    def dispute(self, w0: np.ndarray, w1: np.ndarray) -> Tuple[float, list, int, bool]: raise NotImplementedError
    def witness_inputs(self, witness: list): raise NotImplementedError
    def descriptor(self) -> dict: raise NotImplementedError

    # --- shared ---
    @property
    def tau_value(self) -> float:
        return float(self.tau)

    def normalise_pins(self, pins: Sequence[int]) -> List[int]:
        out = sorted(set(int(p) for p in pins))
        if out and (out[0] < 0 or out[-1] >= self.pin_count):
            raise ValueError("pin outside the host's domain")
        return out

    def forward(self, w, inputs) -> np.ndarray:
        return core.forward(self.shape, w, self.encode(inputs))

    def root(self, w) -> str:
        return core.weight_root(self.shape, w)

    def patterns(self, w, inputs) -> Dict[str, np.ndarray]:
        y = self.forward(w, inputs)
        return {n: self.treadle_pattern(n, inputs, y) for n in self.treadle_names}

    def shed(self, w, pins: Sequence[int]) -> core.Shed:
        pins = self.normalise_pins(pins)
        return core.Shed(self.shape, w, self.encode(self.pin_inputs(pins)) if pins else None)

    def threading(self, w, shed: Optional[core.Shed] = None) -> core.Threading:
        return core.Threading(self.shape, w, self.encode(self.fit_inputs()), shed)

    def witness_change(self, w0, w1, witness: list) -> float:
        """One input is the whole evidence of a dispute: both forward passes, nothing else."""
        return float(core.change(self.shape, w0, w1, self.encode(self.witness_inputs(witness)))[0])


# ---------------------------------------------------------------------------------------
# cloth/1
# ---------------------------------------------------------------------------------------


def grid(n: int) -> np.ndarray:
    """Cell centres of an n x n lattice, row-major: index = row * n + column, u along columns."""
    c = (np.arange(n) + 0.5) / n
    uu, vv = np.meshgrid(c, c)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


class ClothHost(Host):
    kind = "cloth/1"
    INIT_SCALE = {"W1": 1.1, "b1": 0.6, "W2": 0.9, "b2": 0.3, "W3": 0.9, "b3": 0.2}
    FIT_GRID, PIN_LATTICE = 10, 64
    DISPUTE_GRID, DISPUTE_REFINE, DISPUTE_STEPS = 96, 4, 24
    treadle_names = ["warm", "light", "tilt", "contrast", "check"]
    r_invariant, tau = 2.0, "0.5"

    def __init__(self, seed: int = 7):
        self.seed = int(seed)
        self.shape = Shape(8, 16, 3, centre=False)
        self.pin_count = self.PIN_LATTICE ** 2
        draw = core.mulberry32(self.seed)
        parts = []
        for name in core.TENSOR_NAMES:
            lo, hi = self.shape.offsets[name]
            a = self.INIT_SCALE[name]
            # (2u - 1) a uses only exact or correctly rounded operations: the same bytes in JavaScript.
            parts.append(np.array([(2.0 * draw() - 1.0) * a for _ in range(hi - lo)]))
        self.w0 = np.concatenate(parts)

    def encode(self, inputs) -> Enc:
        uv = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        u, v = uv[:, 0], uv[:, 1]
        tau = 2.0 * math.pi
        val = np.stack([2.0 * u - 1.0, 2.0 * v - 1.0, np.sin(tau * u), np.cos(tau * u),
                        np.sin(tau * v), np.cos(tau * v), np.sin(tau * (u + v)), np.sin(tau * (u - v))], axis=1)
        return Enc(np.tile(np.arange(8), (uv.shape[0], 1)), val)

    def fit_inputs(self): return grid(self.FIT_GRID)
    def held_inputs(self): return grid(31)          # no coordinate shared with the fit grid
    def known_inputs(self): return grid(4)
    def elsewhere_inputs(self): return grid(24)

    def pin_inputs(self, pins):
        pins = np.asarray(self.normalise_pins(pins), dtype=np.int64)
        rows, cols = pins // self.PIN_LATTICE, pins % self.PIN_LATTICE
        return np.stack([(cols + 0.5) / self.PIN_LATTICE, (rows + 0.5) / self.PIN_LATTICE], axis=1).reshape(-1, 2)

    def fresh_inputs(self, seed: bytes, n: int):
        return np.array([[core.seeded_uint32(seed, i, 0) / 4294967296.0,
                          core.seeded_uint32(seed, i, 1) / 4294967296.0] for i in range(n)])

    def treadle_pattern(self, name, inputs, y):
        uv = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        n, M = uv.shape[0], self.shape.M
        u, v = uv[:, 0], uv[:, 1]
        ones = np.ones(M)
        if name == "warm":          # first output up, third down
            per = np.tile(np.array([1.0, 0.0, -1.0]), (n, 1))
        elif name == "light":       # all outputs up
            per = np.tile(ones, (n, 1))
        elif name == "tilt":        # up on the right, down on the left
            per = (2.0 * u - 1.0)[:, None] * ones[None, :]
        elif name == "contrast":    # every input away from the mean lightness of the current picture
            light = np.mean(np.asarray(y, dtype=np.float64).reshape(n, M), axis=1)
            per = (light - np.mean(light))[:, None] * ones[None, :]
        elif name == "check":       # a two-period checkerboard
            t = 2.0 * math.pi
            per = (np.sin(2.0 * t * u) * np.sin(2.0 * t * v))[:, None] * ones[None, :]
        else:
            raise ValueError(f"unknown treadle {name!r}")
        return core.unit_rms(per.reshape(n * M))

    def dispute(self, w0, w1):
        """[I] A bounded search: read a whole lattice, then climb from the best cells with a
        shrinking step. A sampled reading: a larger budget can find more."""
        sh = self.shape
        pts = grid(self.DISPUTE_GRID)
        ch = core.change(sh, w0, w1, self.encode(pts))
        rows = int(pts.shape[0])
        order = np.argsort(-ch, kind="stable")[:self.DISPUTE_REFINE]
        best_val, best_pt = float(ch[order[0]]), pts[order[0]].copy()
        for idx in order:
            p, val, step = pts[idx].copy(), float(ch[idx]), 0.5 / self.DISPUTE_GRID
            for _ in range(self.DISPUTE_STEPS):
                cand = np.clip(p[None, :] + step * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 0.0, 1.0)
                vals = core.change(sh, w0, w1, self.encode(cand))
                rows += 4
                j = int(np.argmax(vals))
                if float(vals[j]) > val:
                    p, val = cand[j].copy(), float(vals[j])
                else:
                    step *= 0.5
            if val > best_val:
                best_val, best_pt = val, p
        return best_val, [float(best_pt[0]), float(best_pt[1])], rows, False

    def witness_inputs(self, witness):
        x = np.array([witness], dtype=np.float64)
        if not (np.all(x >= 0.0) and np.all(x <= 1.0)):
            raise ValueError("witness outside the unit square")
        return x

    def descriptor(self):
        return {"kind": self.kind, "seed": self.seed, "w0_root": self.root(self.w0)}


# ---------------------------------------------------------------------------------------
# text/1
# ---------------------------------------------------------------------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyz ."
VOWELS = "aeiou"


def normalise_text(raw: str) -> str:
    """Lower case; letters kept; sentence ends become a full stop; everything else a space;
    runs of spaces collapsed. The 28 symbols of ALPHABET are all that remain."""
    out = []
    for ch in raw.lower():
        if "a" <= ch <= "z":
            out.append(ch)
        elif ch in ".!?":
            out.append(".")
        else:
            out.append(" ")
    text = " ".join("".join(out).split())
    return text.replace(" .", ".")


class TextHost(Host):
    kind = "text/1"
    V, K = 28, 3
    treadle_names = ["vowels", "spaces", "stops", "echo", "sharp"]
    travel = 1.0

    def __init__(self, path: Optional[Path] = None, data: Optional[dict] = None):
        if data is None:
            path = Path(path) if path else REPO / "hosts" / "text-v1.json"
            data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema"] != "heedle-host/text/1" or data["alphabet"] != ALPHABET:
            raise ValueError("not a text/1 host file")
        self.name = data["name"]
        self.shape = Shape(self.K * self.V, int(data["hidden"]), self.V, centre=True)
        self.w0 = np.concatenate([np.asarray(data["weights"][n], dtype=np.float64).ravel() for n in core.TENSOR_NAMES])
        if self.w0.shape != (self.shape.P,):
            raise ValueError("weights do not fit the shape")
        self.r_invariant, self.tau = float(data["r_invariant"]), str(data["tau"])
        self.fit = [int(c) for c in data["contexts"]["fit"]]
        self.known = [int(c) for c in data["contexts"]["known"]]
        self.held = [int(c) for c in data["contexts"]["held"]]
        self.held_counts = [int(c) for c in data["contexts"]["held_counts"]]
        self.held_cumulative = np.cumsum(self.held_counts)
        self.pin_count = self.V ** self.K
        self.recorded_root = data["w0_root"]

    # A context is an integer: c0 * V^2 + c1 * V + c2, with c2 the most recent character.
    def context_of(self, s: str) -> int:
        if len(s) != self.K or any(ch not in ALPHABET for ch in s):
            raise ValueError(f"a context is {self.K} characters of the alphabet")
        c = 0
        for ch in s:
            c = c * self.V + ALPHABET.index(ch)
        return c

    def context_text(self, c: int) -> str:
        return "".join(ALPHABET[(int(c) // self.V ** (self.K - 1 - i)) % self.V] for i in range(self.K))

    def encode(self, inputs) -> Enc:
        c = np.atleast_1d(np.asarray(inputs, dtype=np.int64))
        digits = np.stack([(c // self.V ** (self.K - 1 - i)) % self.V + i * self.V for i in range(self.K)], axis=1)
        return Enc(digits, np.ones(digits.shape))

    def fit_inputs(self): return np.array(self.fit)
    def held_inputs(self): return np.array(self.held)
    def known_inputs(self): return np.array(self.known[:8])   # the eight commonest contexts
    def elsewhere_inputs(self): return np.array(self.held[:200])
    def pin_inputs(self, pins): return np.array(self.normalise_pins(pins), dtype=np.int64)

    def fresh_inputs(self, seed: bytes, n: int):
        """Held-out contexts, drawn in proportion to how often the corpus uses them. Integer
        arithmetic only: a hash word modulo the total count, then a search of the running sum."""
        total = int(self.held_cumulative[-1])
        picks = [int(np.searchsorted(self.held_cumulative, core.seeded_uint32(seed, i) % total, side="right"))
                 for i in range(n)]
        return np.array([self.held[p] for p in picks])

    def treadle_pattern(self, name, inputs, y):
        c = np.atleast_1d(np.asarray(inputs, dtype=np.int64))
        n, M = c.shape[0], self.V
        per = np.zeros((n, M))
        if name == "vowels":        # the five vowels up
            per[:, [ALPHABET.index(ch) for ch in VOWELS]] = 1.0
        elif name == "spaces":      # the space up: shorter words
            per[:, ALPHABET.index(" ")] = 1.0
        elif name == "stops":       # the full stop up: shorter sentences
            per[:, ALPHABET.index(".")] = 1.0
        elif name == "echo":        # the character just written, again: doubled letters
            per[np.arange(n), c % self.V] = 1.0
        elif name == "sharp":       # more of whatever the model already prefers
            per = np.asarray(y, dtype=np.float64).reshape(n, M).copy()
        else:
            raise ValueError(f"unknown treadle {name!r}")
        per = per - per.mean(axis=1, keepdims=True)     # outputs are centred, so patterns are too
        return core.unit_rms(per.reshape(n * M))

    def dispute(self, w0, w1):
        """[P] Every context there is, read once. Exhaustive over the host's domain, in
        float64: the largest change found is the largest there is."""
        everything = np.arange(self.pin_count)
        ch = core.change(self.shape, w0, w1, self.encode(everything))
        worst = int(np.argmax(ch))
        return float(ch[worst]), [worst], int(self.pin_count), True

    def witness_inputs(self, witness):
        c = int(witness[0])
        if not 0 <= c < self.pin_count:
            raise ValueError("witness outside the host's domain")
        return np.array([c])

    def descriptor(self):
        return {"kind": self.kind, "name": self.name, "w0_root": self.root(self.w0)}

    # --- weaving text: what the instrument shows. Not part of any check. ---
    def weave(self, w, tail: str, length: int, seed: int) -> str:
        """Sample `length` characters after `tail`, one uniform draw per character from a
        seeded generator, so two sets of weights are compared on the same noise."""
        draw = core.mulberry32(seed)
        text = tail[-self.K:].rjust(self.K)
        out = []
        for _ in range(length):
            y = core.forward(self.shape, w, self.encode([self.context_of(text[-self.K:])]))[0]
            p = np.exp(y - y.max())
            p /= p.sum()
            ch = ALPHABET[min(self.V - 1, int(np.searchsorted(np.cumsum(p), draw(), side="right")))]
            out.append(ch)
            text += ch
        return "".join(out)


def host_from_descriptor(desc: dict) -> Host:
    """The host a draft names."""
    if desc["kind"] == "cloth/1":
        return ClothHost(int(desc["seed"]))
    if desc["kind"] == "text/1":
        return TextHost(REPO / "hosts" / f"{desc['name']}.json")
    raise ValueError(f"unknown host kind {desc['kind']!r}")
