"""Command line:

    python3 -m heedle verify DRAFT.json     replay a draft and report every difference
    python3 -m heedle hosts                 list the hosts and their baseline roots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import draft
from .hosts import ClothHost, TextHost


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "verify":
        doc = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        problems = draft.replay(doc)
        events = doc.get("events", [])
        count = {k: sum(1 for e in events if e.get("kind") == k) for k in ("pick", "snap", "unpick")}
        host = doc.get("host", {})
        print(f"{host.get('kind')} draft, {len(events)} events: {count['pick']} picks, {count['snap']} snaps, {count['unpick']} unpicks")
        for p in problems:
            print("DIFFERS:", p)
        if not problems:
            print("every root, verdict, witness and reading checks out")
        return 1 if problems else 0
    if len(argv) == 2 and argv[1] == "hosts":
        for host in (ClothHost(7), TextHost()):
            d = host.descriptor()
            print(f"{d['kind']:8s} {host.shape.P:5d} weights  radius {host.r_invariant}  tau {host.tau}  root {d['w0_root']}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
