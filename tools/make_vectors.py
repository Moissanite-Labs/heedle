#!/usr/bin/env python3
"""Write vectors/cloth.json and vectors/text.json: values computed by the reference
(heedle/) that the JavaScript mirror (web/) must reproduce.

    python3 tools/make_vectors.py

Exact entries (hashes, generator draws, probes) must match bit for bit. Numeric entries
carry the tolerance the JavaScript test applies; they differ only through the last digits
of tanh and sin, and through block iteration against an exact SVD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from heedle import core  # noqa: E402
from heedle.hosts import ClothHost, TextHost  # noqa: E402


def test_move(P: int, scale: float) -> np.ndarray:
    """A fixed move built from integers only, so both languages hold the same bytes."""
    return np.array([(((i * 37) % 101) - 50) / 50.0 for i in range(P)]) * scale


def test_pattern(n: int) -> np.ndarray:
    return np.array([(((j * 29) % 53) - 26) / 26.0 for j in range(n)])


def vectors(host, pins, scale, some_inputs, hide_radius) -> dict:
    sh, w = host.shape, host.w0
    d = test_move(sh.P, scale)
    fit_inputs = host.fit_inputs()
    fit = host.encode(fit_inputs)
    some = host.encode(some_inputs)
    th = host.threading(w)
    tie = core.TieUp(th, host.patterns(w, fit_inputs), 16)
    shed = host.shed(w, pins)
    th_shed = host.threading(w, shed)
    tie_shed = core.TieUp(th_shed, host.patterns(w, fit_inputs), 16)
    w1 = w + d
    root0, root1 = host.root(w), host.root(w1)
    cid = core.claim_id(host.kind, root0, root1, host.tau, pins)
    fresh = host.fresh_inputs(core.probe_seed(root1, cid), 8)
    big = w + th.shaft_move(0, 0.4)
    worst, witness, rows, exhaustive = host.dispute(w, big)
    known, elsewhere = host.encode(host.known_inputs()), host.encode(host.elsewhere_inputs())
    wh = w + core.hidden_move(sh, w, known, elsewhere, hide_radius)
    r = test_pattern(some.n * sh.M)
    draw = core.mulberry32(12345)
    return {
        "schema": "heedle-vectors/1",
        "note": "Written by tools/make_vectors.py from heedle/. Do not edit by hand.",
        "host": host.descriptor(),
        "pins": [int(p) for p in pins],
        "move_scale": scale,
        "inputs": np.asarray(some_inputs).tolist(),
        "hide_radius": hide_radius,
        "exact": {
            "mulberry32_12345": [draw() for _ in range(8)],
            "weights": int(sh.P),
            "weights_head": w[:4].tolist(),
            "w0_root": root0,
            "w1_root": root1,
            "claim_id": cid,
            "fresh_inputs": np.asarray(fresh).tolist(),
            "event_id": core.event_id({"seq": 1, "kind": "pick", "prev": "genesis", "parent_root": root0, "root": root1,
                                       "move_sha256": "00" * 32, "pins": [int(p) for p in pins], "gesture": "hold", "tau": host.tau}),
            "open_dimensions": int(shed.open_dimensions),
            "dispute_rows": int(rows),
            "dispute_exhaustive": bool(exhaustive),
        },
        "numeric": {
            "forward": {"values": core.forward(sh, w, some).ravel().tolist(), "atol": 1e-11},
            "jvp": {"values": core.jvp(sh, w, some, d).ravel().tolist(), "atol": 1e-12},
            "vjp_dot_move": {"value": float(core.vjp(sh, w, some, r.reshape(some.n, sh.M)) @ d), "atol": 1e-11},
            "eigenvalues_16": {"values": th.lam[:16].tolist(), "rtol": 1e-6},
            "predicted_change": {"value": th.predicted_change(d), "rtol": 1e-9},
            "share_16": {"values": tie.share, "atol": 1e-5},
            "unit_move_length_16": {"values": {k: float(np.linalg.norm(v)) for k, v in tie.unit_move.items()}, "rtol": 1e-4},
            "eigenvalues_16_in_shed": {"values": th_shed.lam[:16].tolist(), "rtol": 1e-6},
            "share_16_in_shed": {"values": tie_shed.share, "atol": 1e-5},
            "projected_move_length": {"value": float(np.linalg.norm(shed.project(d))), "rtol": 1e-9},
            "certified_bound": {"radius": 0.3, "values": core.certified_bound(sh, w, some, 0.3).tolist(), "rtol": 1e-9},
            "behaviour_change": {"value": core.behaviour_change(sh, w, w1, fit), "rtol": 1e-8},
            "dispute": {"travel": 0.4, "worst": worst, "witness": witness, "rtol": 1e-6},
            "hidden_move": {"on_known": core.behaviour_change(sh, w, wh, known),
                            "elsewhere": core.behaviour_change(sh, w, wh, elsewhere), "rtol": 1e-3},
        },
    }


def main() -> None:
    cloth = ClothHost(7)
    text = TextHost()
    out = {
        "cloth": vectors(cloth, [100, 2000, 3000], 0.01, [[0.1, 0.2], [0.5, 0.5], [0.9, 0.05], [0.33, 0.77], [0.0, 1.0]], 0.5),
        "text": vectors(text, [text.context_of(" th"), text.context_of("the")], 0.002,
                        [text.context_of(s) for s in (" th", "ing", "qzx", "e. ", "...")], 0.5),
    }
    for name, doc in out.items():
        path = REPO / "vectors" / f"{name}.json"
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
