#!/usr/bin/env python3
"""Stage 2: split products by model family, per type: train 50 / dev 10 / calibration 25 / test 15.

A model family is one product sold in several colours or sizes (e.g. "Rivet Julie dining
chair" in grey and in navy). The whole family goes to one split, so no test product has a
near-copy in training. Splitting by brand alone failed: three Amazon house brands make up
about 90% of the furniture and lamps, so the splits came out lopsided (lamps: 6% test).

Within each type, families are taken largest first (ties in a fixed random order) and each
goes to the split furthest below its target share.

Run: .venv-serve/bin/python scripts/stage2/make_splits.py   -> data/splits.json
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from catalog_cascade import schema

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"train": 0.50, "dev": 0.10, "calibration": 0.25, "test": 0.15}
COLOUR_WORDS = set(schema.COLORS) | set(schema.SYNONYMS["color"]) | {"dark", "light", "ash"}


def family(product: dict) -> str:
    """Brand plus the first 6 words of the title's name part, colour words removed.

    "Amazon Brand – Rivet Julie Dining Chair, 32"H, Ash Grey" -> "rivet|rivet julie dining
    chair": the part after the first comma or dash (size, colour) is dropped.
    """
    title = product["title"].lower()
    title = re.sub(r"^amazon (brand|basics|commercial)\s*[-–:]?\s*", "", title)
    name = re.split(r",| - | – |\(|\|", title)[0]
    words = [w for w in re.sub(r"[^a-z ]", " ", name).split() if w not in COLOUR_WORDS]
    return f"{product['brand']}|{' '.join(words[:6])}"


def main() -> int:
    products = [json.loads(line) for line in (ROOT / "data" / "products.jsonl").open()]
    by_type: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p in products:
        by_type[p["product_type"]][family(p)].append(p["item_id"])

    splits: dict[str, str] = {}
    for ptype, families in sorted(by_type.items()):
        total = sum(len(ids) for ids in families.values())
        filled = Counter()
        order = sorted(families.items())
        random.Random(0).shuffle(order)            # fixed random order among equal sizes
        order.sort(key=lambda kv: -len(kv[1]))     # then largest first (sort keeps ties)
        for _, ids in order:
            # the split with the largest gap between its target and what it holds so far
            split = max(TARGETS, key=lambda s: TARGETS[s] * total - filled[s])
            filled[split] += len(ids)
            for item_id in ids:
                splits[item_id] = split
        print(f"{ptype:14s} {len(families):5d} families",
              {s: round(filled[s] / total, 2) for s in TARGETS}, flush=True)

    (ROOT / "data" / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    print("Total:", dict(Counter(splits.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
