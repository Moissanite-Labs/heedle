#!/usr/bin/env python3
"""Train the text host and write hosts/<name>.json and hosts/<name>.corpus.txt.

    python3 tools/train_text_host.py --name text-v1

The corpus is this repository's own prose: the charter, the contributing and reviewing
guides, and docs/*.md, with fenced code removed, reduced to 28 symbols. The script counts how
often each context of three characters is followed by each character and fits the three-layer
network to those counts by Adam on the cross-entropy.

The result depends on the documents as they were when the script ran, and its last digits
on the NumPy build. So the output is an artefact with a name: the corpus file fixes the
text, the host file records its hash and its own weight root, and a retrained model gets a
new name. Drafts made on an old model keep verifying against the old file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from heedle import core  # noqa: E402
from heedle.hosts import ALPHABET, normalise_text  # noqa: E402

V, K, HIDDEN = 28, 3, 32
SOURCES = ["CHARTER.md", "CONTRIBUTING.md", "REVIEWING.md"]
INIT = {"W1": 0.5, "b1": 0.1, "W2": 0.4, "b2": 0.1, "W3": 0.4, "b3": 0.1}
STEPS, WEIGHT_DECAY = 6000, 2e-4
FIT, KNOWN, HELD = 48, 16, 512
R_INVARIANT, TAU = "2.0", "1.0"


def corpus_text() -> tuple[str, list[dict]]:
    files = [REPO / s for s in SOURCES] + sorted((REPO / "docs").glob("*.md"))
    parts, record = [], []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        record.append({"path": str(f.relative_to(REPO)), "sha256": hashlib.sha256(raw.encode()).hexdigest()})
        parts.append(re.sub(r"```.*?```", " ", raw, flags=re.S))
    return normalise_text(" . ".join(parts)), record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="text-v1")
    args = ap.parse_args()
    text, sources = corpus_text()
    corpus_file = REPO / "hosts" / f"{args.name}.corpus.txt"
    corpus_file.write_text("\n".join(textwrap.wrap(text, 100, break_long_words=False)) + "\n", encoding="utf-8")
    text = " ".join(corpus_file.read_text(encoding="utf-8").split("\n")).strip()   # train on what was written
    codes = np.array([ALPHABET.index(ch) for ch in text])
    ctx = codes[:-3] * V * V + codes[1:-2] * V + codes[2:-1]
    counts = np.zeros((V ** K, V))
    np.add.at(counts, (ctx, codes[3:]), 1.0)
    totals = counts.sum(axis=1)
    seen = np.flatnonzero(totals)
    N = float(totals.sum())

    sh = core.Shape(K * V, HIDDEN, V, centre=True)
    draw = core.mulberry32(1)
    w = np.concatenate([np.array([(2.0 * draw() - 1.0) * INIT[n] for _ in range(hi - lo)])
                        for n, (lo, hi) in ((n, sh.offsets[n]) for n in core.TENSOR_NAMES)])
    digits = np.stack([(seen // V ** (K - 1 - i)) % V + i * V for i in range(K)], axis=1)
    enc = core.Enc(digits, np.ones(digits.shape))
    q = counts[seen] / totals[seen, None]
    weight = totals[seen] / N

    def loss_and_grad(w):
        y = core.forward(sh, w, enc)
        y = y - y.max(axis=1, keepdims=True)
        p = np.exp(y)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-(weight[:, None] * q * np.log(p + 1e-300)).sum())
        return loss, core.vjp(sh, w, enc, weight[:, None] * (p - q)) + WEIGHT_DECAY * w

    m, v = np.zeros(sh.P), np.zeros(sh.P)
    first = None
    for t in range(1, STEPS + 1):
        loss, g = loss_and_grad(w)
        first = first if first is not None else loss
        lr = 0.02 if t <= STEPS // 2 else 0.005
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0:
            print(f"step {t}: {loss / np.log(2):.4f} bits per character", flush=True)
    loss, _ = loss_and_grad(w)
    unigram = counts.sum(axis=0) / N
    by_count = seen[np.argsort(-totals[seen], kind="stable")]
    host = {
        "schema": "heedle-host/text/1",
        "name": args.name,
        "alphabet": ALPHABET,
        "context": K,
        "hidden": HIDDEN,
        "r_invariant": R_INVARIANT,
        "tau": TAU,
        "weights": {n: sh.tensor(w, n).tolist() for n in core.TENSOR_NAMES},
        "w0_root": core.weight_root(sh, w),
        "contexts": {
            "fit": [int(c) for c in by_count[:FIT]],
            "known": [int(c) for c in by_count[:KNOWN]],
            "held": [int(c) for c in by_count[FIT:FIT + HELD]],
            "held_counts": [int(totals[c]) for c in by_count[FIT:FIT + HELD]],
        },
        "training": {
            "corpus": {"path": f"hosts/{args.name}.corpus.txt", "sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
                       "characters": int(len(text)), "contexts_seen": int(len(seen))},
            "sources": sources,
            "steps": STEPS, "weight_decay": WEIGHT_DECAY,
            "bits_per_character_before": first / float(np.log(2)),
            "bits_per_character_after": loss / float(np.log(2)),
            "bits_per_character_unigram": float(-(unigram * np.log2(unigram + 1e-300)).sum()),
            "python": platform.python_version(), "numpy": np.__version__,
        },
    }
    out = REPO / "hosts" / f"{args.name}.json"
    out.write_text(json.dumps(host) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}: root {host['w0_root'][:16]}, {len(text)} characters, {len(seen)} contexts seen")
    print(f"bits per character: {host['training']['bits_per_character_before']:.3f} before, "
          f"{host['training']['bits_per_character_after']:.3f} after, unigram {host['training']['bits_per_character_unigram']:.3f}")


if __name__ == "__main__":
    main()
