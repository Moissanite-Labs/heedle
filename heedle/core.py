"""Heedle core: the mathematics of steering a small model by hand, for any host.

This is the reference. `web/heedle.js` mirrors it, and `vectors/` pins the two together.
`docs/theory.md` gives the derivations and says which statements are proved and which are
only tested; `docs/vocabulary.md` says what each weaving word names here.

A *host* is a three-layer network

    f(x, w) = W3 g(W2 g(W1 x + b1) + b2) + b3,        g = tanh  (slope at most 1),

with a fixed way of turning its inputs into feature vectors x. It is the network of Theorem 1
in weft-loom's calibration theory, so that theorem's proved bound applies to every host.
Feature vectors are sparse: an encoded batch holds, per input, `nnz` feature indices and
their values. A host may centre its outputs (subtract their mean), which a model that ends
in a softmax does not notice.

Nothing in this file knows what a host's inputs mean. `heedle/hosts.py` does.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TENSOR_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]
GAMMA = 1.0                       # largest slope of tanh
CBC_DOMAIN = b"WEFT-CBC-V1:"      # the weight-root domain of committed-behaviour-change v1
DOMAIN = b"HEEDLE-V1:"            # everything else hashed here

# ---------------------------------------------------------------------------------------
# Deterministic numbers: the same bits in Python and JavaScript
# ---------------------------------------------------------------------------------------


def mulberry32(seed: int) -> Callable[[], float]:
    """The mulberry32 generator. Integer arithmetic only, so every draw is bit-identical in
    JavaScript and Python. Returns a function giving a float in [0, 1)."""
    state = [seed & 0xFFFFFFFF]

    def imul(a: int, b: int) -> int:
        return (a * b) & 0xFFFFFFFF

    def draw() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        a = state[0]
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return draw


def _only_strings_and_integers(obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError("a hashed object has a key that is not a string")
            _only_strings_and_integers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _only_strings_and_integers(v)
    elif isinstance(obj, bool) or not isinstance(obj, (str, int)):
        raise ValueError(f"only strings and integers are hashed, not {obj!r}")


def canonical_json(obj) -> bytes:
    """JSON with sorted keys and no spaces, of strings, integers and lists of them only.
    Anything else is refused: a fraction prints differently in different languages."""
    _only_strings_and_integers(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------
# Shape and encoded inputs
# ---------------------------------------------------------------------------------------


class Shape:
    """Sizes of a host: F features, H hidden units (both layers), M outputs."""

    def __init__(self, F: int, H: int, M: int, centre: bool = False):
        self.F, self.H, self.M, self.centre = int(F), int(H), int(M), bool(centre)
        self.shapes: Dict[str, Tuple[int, ...]] = {
            "W1": (H, F), "b1": (H,), "W2": (H, H), "b2": (H,), "W3": (M, H), "b3": (M,)}
        self.offsets: Dict[str, Tuple[int, int]] = {}
        at = 0
        for name in TENSOR_NAMES:
            size = int(np.prod(self.shapes[name]))
            self.offsets[name] = (at, at + size)
            at += size
        self.P = at

    def tensor(self, w: np.ndarray, name: str) -> np.ndarray:
        lo, hi = self.offsets[name]
        return w[lo:hi].reshape(self.shapes[name])

    def tensors(self, w: np.ndarray) -> Dict[str, np.ndarray]:
        return {n: self.tensor(w, n) for n in TENSOR_NAMES}


@dataclass
class Enc:
    """Encoded inputs: per input, `nnz` feature indices and values. Indices within one
    input are distinct."""
    idx: np.ndarray   # (n, nnz) integers
    val: np.ndarray   # (n, nnz) floats

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])


# ---------------------------------------------------------------------------------------
# The host network: forward pass, directional derivative, its transpose, and the Jacobian
# ---------------------------------------------------------------------------------------


def _layers(sh: Shape, w: np.ndarray, enc: Enc):
    W1, b1, W2, b2 = (sh.tensor(w, n) for n in ("W1", "b1", "W2", "b2"))
    a1 = np.einsum("hnk,nk->nh", W1[:, enc.idx], enc.val) + b1
    h1 = np.tanh(a1)
    h2 = np.tanh(h1 @ W2.T + b2)
    return h1, h2


def _centred(sh: Shape, y: np.ndarray) -> np.ndarray:
    return y - y.mean(axis=1, keepdims=True) if sh.centre else y


def forward(sh: Shape, w: np.ndarray, enc: Enc) -> np.ndarray:
    """Outputs, shape (n, M)."""
    _, h2 = _layers(sh, w, enc)
    return _centred(sh, h2 @ sh.tensor(w, "W3").T + sh.tensor(w, "b3"))


def jvp(sh: Shape, w: np.ndarray, enc: Enc, d: np.ndarray) -> np.ndarray:
    """[K] J d: the first-order output change of a weight move d, shape (n, M)."""
    h1, h2 = _layers(sh, w, enc)
    W2, W3 = sh.tensor(w, "W2"), sh.tensor(w, "W3")
    dW1, db1, dW2, db2, dW3, db3 = (sh.tensor(d, n) for n in TENSOR_NAMES)
    dh1 = (1.0 - h1 * h1) * (np.einsum("hnk,nk->nh", dW1[:, enc.idx], enc.val) + db1)
    dh2 = (1.0 - h2 * h2) * (dh1 @ W2.T + h1 @ dW2.T + db2)
    return _centred(sh, dh2 @ W3.T + h2 @ dW3.T + db3)


def vjp(sh: Shape, w: np.ndarray, enc: Enc, R: np.ndarray) -> np.ndarray:
    """[K] J' r: the weight direction that an output pattern R, shape (n, M), pulls on."""
    h1, h2 = _layers(sh, w, enc)
    W2, W3 = sh.tensor(w, "W2"), sh.tensor(w, "W3")
    R = _centred(sh, np.asarray(R, dtype=np.float64).reshape(enc.n, sh.M))
    d2 = (R @ W3) * (1.0 - h2 * h2)
    d1 = (d2 @ W2) * (1.0 - h1 * h1)
    gW1T = np.zeros((sh.F, sh.H))
    np.add.at(gW1T, enc.idx, d1[:, None, :] * enc.val[:, :, None])
    return np.concatenate([gW1T.T.ravel(), d1.sum(0), (d2.T @ h1).ravel(), d2.sum(0),
                           (R.T @ h2).ravel(), R.sum(0)])


def jacobian(sh: Shape, w: np.ndarray, enc: Enc) -> np.ndarray:
    """[K] Derivatives of every output on every input with respect to every weight.
    Shape (n * M, P), row = input * M + output. Dense: for small input sets and for tests."""
    n = enc.n
    h1, h2 = _layers(sh, w, enc)
    W2, W3 = sh.tensor(w, "W2"), sh.tensor(w, "W3")
    g1, g2 = 1.0 - h1 * h1, 1.0 - h2 * h2
    J = np.zeros((n, sh.M, sh.P))
    o = sh.offsets
    rows = np.arange(n)
    for c in range(sh.M):
        r = np.zeros(sh.M)
        r[c] = 1.0
        if sh.centre:
            r -= 1.0 / sh.M
        d2 = (r @ W3)[None, :] * g2
        d1 = (d2 @ W2) * g1
        block = np.zeros((n, sh.H, sh.F))
        for t in range(enc.idx.shape[1]):
            block[rows, :, enc.idx[:, t]] += d1 * enc.val[:, t, None]
        J[:, c, o["W1"][0]:o["W1"][1]] = block.reshape(n, -1)
        J[:, c, o["b1"][0]:o["b1"][1]] = d1
        J[:, c, o["W2"][0]:o["W2"][1]] = (d2[:, :, None] * h1[:, None, :]).reshape(n, -1)
        J[:, c, o["b2"][0]:o["b2"][1]] = d2
        J[:, c, o["W3"][0]:o["W3"][1]] = (r[None, :, None] * h2[:, None, :]).reshape(n, -1)
        J[:, c, o["b3"][0]:o["b3"][1]] = r
    return J.reshape(n * sh.M, sh.P)


def change(sh: Shape, w0: np.ndarray, w1: np.ndarray, enc: Enc) -> np.ndarray:
    """Per-input change: root mean square over the outputs of f1 - f0. The CBC `change(x)`."""
    d = forward(sh, w1, enc) - forward(sh, w0, enc)
    return np.sqrt(np.mean(d * d, axis=1))


def behaviour_change(sh: Shape, w0: np.ndarray, w1: np.ndarray, enc: Enc) -> float:
    """D_X of the calibration theory: root mean square of all output differences on X."""
    d = forward(sh, w1, enc) - forward(sh, w0, enc)
    return float(np.sqrt(np.mean(d * d)))


def predicted_worst(sh: Shape, w: np.ndarray, enc: Enc, d: np.ndarray) -> float:
    """First-order change of the worst single input: the largest per-input RMS of J d."""
    y = jvp(sh, w, enc, d)
    return float(np.max(np.sqrt(np.mean(y * y, axis=1))))


def certified_bound(sh: Shape, w0: np.ndarray, enc: Enc, radius: float) -> np.ndarray:
    """[K] Theorem 1 of the calibration theory, with slope 1 for tanh. For every move of
    length at most `radius` and every input x,
        ||f(x, w0 + d) - f(x, w0)||  <=  R c(x) + R^2 q(x) + R^3 k(x).
    Returns that bound per input, on the Euclidean length of the output change. Centring
    the outputs is a projection, so the bound holds for centred outputs too."""
    h1, h2 = _layers(sh, w0, enc)
    s2 = float(np.linalg.norm(sh.tensor(w0, "W2"), 2))
    s3 = float(np.linalg.norm(sh.tensor(w0, "W3"), 2))
    a0 = np.sqrt(np.sum(enc.val * enc.val, axis=1) + 1.0)
    a1 = np.sqrt(np.sum(h1 * h1, axis=1) + 1.0)
    a2 = np.sqrt(np.sum(h2 * h2, axis=1) + 1.0)
    g, r = GAMMA, float(radius)
    c = np.sqrt((g * g * s3 * s2 * a0) ** 2 + (g * s3 * a1) ** 2 + a2 ** 2)
    q = (g * g * (s2 + s3) * a0 + g * a1) / 2.0
    k = g * g * a0 / (3.0 * math.sqrt(3.0))
    return r * c + r * r * q + r ** 3 * k


# ---------------------------------------------------------------------------------------
# Pins and the shed
# ---------------------------------------------------------------------------------------


def row_space_basis(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal rows spanning the rows of A, by modified Gram-Schmidt run twice. A row
    is dropped when what is left of it is below rtol times its own length. This is the
    algorithm the JavaScript mirror runs; numpy's QR is only the test's oracle."""
    basis: List[np.ndarray] = []
    for row in np.asarray(A, dtype=np.float64):
        scale = float(np.linalg.norm(row))
        if scale == 0.0:
            continue
        v = row.copy()
        for _ in range(2):
            for q in basis:
                v -= float(q @ v) * q
        norm = float(np.linalg.norm(v))
        if norm > rtol * scale:
            basis.append(v / norm)
    return np.array(basis).reshape(len(basis), A.shape[1])


class Shed:
    """[P] The moves that leave the pinned inputs alone to first order: the null space of
    the pins' Jacobian. `project` removes the pins' row space from a move."""

    def __init__(self, sh: Shape, w: np.ndarray, pin_enc: Optional[Enc]):
        if pin_enc is not None and pin_enc.n:
            self.J = jacobian(sh, w, pin_enc)
            self.Q = row_space_basis(self.J)
        else:
            self.J = np.zeros((0, sh.P))
            self.Q = np.zeros((0, sh.P))
        self.open_dimensions = sh.P - self.Q.shape[0]

    def project(self, d: np.ndarray) -> np.ndarray:
        if self.Q.shape[0] == 0:
            return np.array(d, dtype=np.float64, copy=True)
        return d - self.Q.T @ (self.Q @ d)


# ---------------------------------------------------------------------------------------
# Shafts, treadles and the tie-up
# ---------------------------------------------------------------------------------------


class Threading:
    """The linearisation of a host at one set of weights, on its fit probes.

    J = U S V' with singular values s. G = J'J/(n m) has eigenvalues lam = s^2/(n m) on the
    columns of V: the shafts. `U * sqrt(n m)` are the shaft patterns: the unit-RMS output
    pattern each shaft makes. Signs are fixed so the largest entry of each shaft is positive.

    With a shed the loom is threaded around the pins: the shafts are those of G restricted
    to the shed, so every shaft already leaves the pinned inputs alone to first order.
    This reference takes an exact SVD of the dense Jacobian. The JavaScript mirror iterates
    without forming it.
    """

    def __init__(self, sh: Shape, w: np.ndarray, enc: Enc, shed: Optional[Shed] = None):
        self.sh, self.w, self.enc = sh, np.array(w, dtype=np.float64), enc
        self.J = jacobian(sh, w, enc)
        self.rows = self.J.shape[0]
        inside = self.J
        if shed is not None and shed.Q.shape[0]:
            inside = self.J - (self.J @ shed.Q.T) @ shed.Q
        U, s, Vt = np.linalg.svd(inside, full_matrices=False)
        flip = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        flip[flip == 0] = 1.0
        self.U = U * flip[None, :]
        self.V = (Vt * flip[:, None]).T
        self.s = s
        self.lam = s * s / self.rows

    def predicted_change(self, d: np.ndarray) -> float:
        """[P] First-order behaviour change sqrt(d'Gd) of a move on the fit probes."""
        y = self.J @ d
        return float(np.sqrt(np.mean(y * y)))

    def shaft_move(self, i: int, travel: float) -> np.ndarray:
        """[P] The move along shaft i whose first-order behaviour change is |travel|."""
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < len(self.lam):
            raise ValueError(f"shaft index {i!r} out of range")
        idx = int(i)
        lam = float(self.lam[idx])
        if not (math.isfinite(lam) and lam > 0.0):
            raise ValueError(f"shaft {idx} has zero eigenvalue")
        return travel * self.V[:, idx] / math.sqrt(lam)

    def rank(self, rtol: float = 1e-10) -> int:
        return int(np.sum(self.s > rtol * self.s[0]))


class TieUp:
    """Treadles tied to the first K shafts of a threading.

    For a pattern c of unit root mean square, alpha_i = u_i'c / sqrt(n m). With travel t the
    move is  d = t * sum_i alpha_i v_i / sqrt(lam_i), over i < K. [P] To first order the
    output change is t times the projection of c on the K shafts, so its gain along c is
    t * share, with share = sum_i alpha_i^2.
    """

    def __init__(self, threading: Threading, patterns: Dict[str, np.ndarray], shafts: int):
        self.K = int(min(shafts, threading.rank()))
        self.names = list(patterns)
        root = math.sqrt(threading.rows)
        self.patterns = {k: np.asarray(v, dtype=np.float64) for k, v in patterns.items()}
        self.alpha = {k: (threading.U[:, :self.K].T @ c) / root for k, c in self.patterns.items()}
        self.share = {k: float(a @ a) for k, a in self.alpha.items()}
        lam = threading.lam[:self.K]
        self.unit_move = {k: threading.V[:, :self.K] @ (a / np.sqrt(lam)) for k, a in self.alpha.items()}

    def move(self, treadling: Dict[str, float]) -> np.ndarray:
        d = np.zeros(next(iter(self.unit_move.values())).shape[0])
        for name, t in treadling.items():
            d = d + float(t) * self.unit_move[name]
        return d

    def matrix(self) -> np.ndarray:
        """The tie-up, shape (K, number of treadles)."""
        return np.stack([self.alpha[k] for k in self.names], axis=1)


def unit_rms(c: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(c * c)))
    return np.zeros_like(c) if r < 1e-300 else c / r


def gain_along(dy: np.ndarray, pattern: np.ndarray) -> float:
    """How much of `pattern` an output change contains: <dy, c> / <c, c>."""
    return float(np.dot(np.ravel(dy), pattern) / np.dot(pattern, pattern))


def hidden_move(sh: Shape, w: np.ndarray, known: Enc, elsewhere: Enc, radius: float,
                iterations: int = 60) -> np.ndarray:
    """[K] A move of length `radius` that the known probes cannot see to first order and
    that changes as much as it can elsewhere: power iteration on the elsewhere Jacobian
    inside the null space of the known probes' Jacobian, from a fixed start."""
    Q = row_space_basis(jacobian(sh, w, known))
    draw = mulberry32(20261117)
    v = np.array([2.0 * draw() - 1.0 for _ in range(sh.P)])

    def project(z: np.ndarray) -> np.ndarray:
        z = z - Q.T @ (Q @ z)
        return z / np.linalg.norm(z)

    v = project(v)
    for _ in range(iterations):
        v = project(vjp(sh, w, elsewhere, jvp(sh, w, elsewhere, v)))
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return float(radius) * v


# ---------------------------------------------------------------------------------------
# The selvedge: weft's gate
# ---------------------------------------------------------------------------------------


def verify_weight_update_safety(baseline: Iterable[float], candidate: Iterable[float],
                                r_invariant: float, budget: float, cost: float
                                ) -> Tuple[bool, float, float, str]:
    """A port of the unmasked list path of `verify_weight_update_safety_ex` in weft's
    `weft/runtime/invariants/invariants.py`, checked in the tests against weft's own
    conformance vectors. Returns (accepted, distance, remaining_budget, reason).

    The gate accepts when  budget >= cost > 0  and  ||candidate - baseline|| <= r_invariant.
    """
    b, c, r = float(budget), float(cost), float(r_invariant)
    if not (math.isfinite(b) and math.isfinite(c) and math.isfinite(r)):
        return False, math.inf, math.nan, "non_finite_parameters"
    if c <= 0.0:
        return False, math.inf, b, "non_positive_budget_or_cost"
    if b <= 0.0 or r < 0.0:
        return False, math.inf, b - c, "non_positive_budget_or_cost"
    if b < c:
        return False, math.inf, b - c, "insufficient_metabolic_budget"
    base = [float(v) for v in np.asarray(baseline, dtype=np.float64).ravel()]
    cand = [float(v) for v in np.asarray(candidate, dtype=np.float64).ravel()]
    if len(base) != len(cand) or len(base) == 0:
        return False, math.inf, b, "shape_mismatch"
    sq = 0.0
    for x, y in zip(base, cand):
        if not (math.isfinite(x) and math.isfinite(y)):
            return False, math.inf, b, "non_finite_weights"
        diff = y - x
        sq += diff * diff
    dist = math.sqrt(sq)
    if not math.isfinite(dist):
        return False, math.inf, b, "non_finite_weights"
    if dist > r:
        return False, dist, b, "baseline_distance_exceeded"
    return True, dist, b - c, "ok"


# ---------------------------------------------------------------------------------------
# Commitments
# ---------------------------------------------------------------------------------------


def weight_root(sh: Shape, w: np.ndarray) -> str:
    """The weight root of committed-behaviour-change v1 (`cbc.py::weight_root` in weft-loom)
    over the host's six named tensors, float64."""
    h = hashlib.sha256(CBC_DOMAIN + b"root:")
    for name in sorted(TENSOR_NAMES):
        a = np.ascontiguousarray(sh.tensor(np.asarray(w, dtype="<f8"), name))
        header = canonical_json({"name": name, "shape": list(a.shape), "dtype": "<f8"})
        leaf = sha256_hex(CBC_DOMAIN + b"leaf:" + header + b":" + a.tobytes())
        h.update(name.encode() + b":" + leaf.encode() + b";")
    return h.hexdigest()


def claim_id(kind: str, w0_root: str, w1_root: str, tau: str, pins: Sequence[int]) -> str:
    """Identifier of the claim a pick makes: no input of the host's domain changes by more
    than tau. Strings and integers only, so both languages hash the same bytes."""
    body = {"schema": "heedle-claim/1", "host": kind, "w0_root": w0_root, "w1_root": w1_root,
            "tau": str(tau), "pins": sorted(set(int(p) for p in pins))}
    return sha256_hex(DOMAIN + b"claim:" + canonical_json(body))


def probe_seed(w1_root: str, claim: str, test_set: str = "admission") -> bytes:
    """[K] The seed of the probes drawn after a candidate is fixed, as CBC v1 orders it: it
    hashes the candidate's root and the claim, so the probes of a candidate are fixed only
    once the candidate is. A proposer can still compute the probes of any candidate it
    considers, which is one reason probes measure a pick and never decide one."""
    return hashlib.sha256(DOMAIN + b"probes:" + test_set.encode() + b":" + w1_root.encode()
                          + b":" + claim.encode()).digest()


def seeded_uint32(seed: bytes, i: int, j: int = 0) -> int:
    """The j-th 32-bit integer of the i-th hash of a probe seed."""
    h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
    return int.from_bytes(h[4 * j:4 * j + 4], "big")


def move_to_b64(d: np.ndarray) -> str:
    return base64.b64encode(np.asarray(d, dtype="<f8").tobytes()).decode()


def move_from_b64(s: str, P: int) -> np.ndarray:
    d = np.frombuffer(base64.b64decode(s), dtype="<f8").astype(np.float64)
    if d.shape != (P,):
        raise ValueError("a move must hold one float64 per weight")
    return d


def genesis_id(host: dict, gate: dict) -> str:
    """What the first event of a draft chains from: the hash of the draft's header. It binds
    every event to the host and to the session's limits."""
    return sha256_hex(DOMAIN + b"draft:" + canonical_json({"schema": "heedle-draft/1", "host": host, "gate": gate}))


def event_id(event: dict) -> str:
    """Identifier of a draft event. It covers the exact commitments only (strings and
    integers); the readings in an event are recomputed by a verifier, not trusted."""
    body = {k: event[k] for k in ("seq", "kind", "prev", "parent_root", "root", "move_sha256",
                                  "pins", "gesture", "tau") if k in event}
    return sha256_hex(DOMAIN + b"event:" + canonical_json(body))


ZONES = ["auto", "shared", "knot"]
GESTURES = ["release", "hold", "knot"]


def zone_of(tension: float) -> str:
    """Autonomy zones, with the thresholds of the governor layer in the corpus's genesis
    dialogue: below 0.3, from 0.3, from 0.7."""
    if tension < 0.3:
        return "auto"
    if tension < 0.7:
        return "shared"
    return "knot"
