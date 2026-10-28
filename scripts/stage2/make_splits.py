#!/usr/bin/env python3
"""Stage 2: split products by brand family, per type: train 50 / dev 10 / calibration 25 / test 15.

A whole brand goes to one split, so the test set holds brands the model never trained on.
Within each type, families are taken largest first and each goes to the split furthest
below its target share. Products with no brand count as a family of one.

Run: .venv-serve/bin/python scripts/stage2/make_splits.py   -> data/splits.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"train": 0.50, "dev": 0.10, "calibration": 0.25, "test": 0.15}


def main() -> int:
    products = [json.loads(line) for line in (ROOT / "data" / "products.jsonl").open()]
    by_type: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p in products:
        family = p["brand"] or f"no-brand:{p['item_id']}"
        by_type[p["product_type"]][family].append(p["item_id"])

    splits: dict[str, str] = {}
    for ptype, families in sorted(by_type.items()):
        total = sum(len(ids) for ids in families.values())
        filled = Counter()
        order = sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        random.Random(0).shuffle(order[len(order) // 10:])  # keep the big ones first
        for _, ids in order:
            # the split with the largest gap between its target and what it holds so far
            split = max(TARGETS, key=lambda s: TARGETS[s] * total - filled[s])
            filled[split] += len(ids)
            for item_id in ids:
                splits[item_id] = split
        print(ptype, {s: round(filled[s] / total, 2) for s in TARGETS}, flush=True)

    (ROOT / "data" / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    print("Total:", dict(Counter(splits.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
