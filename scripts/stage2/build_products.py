#!/usr/bin/env python3
"""Stage 2: the product list for the six types, and their photos at full size.

Writes data/products.jsonl (one product per line: id, type, title, brand, image files,
seller values) and data/images/<image_id>.jpg, shrunk so the longer side is 768 px
(Tier 2's size; Tier 1 shrinks again to 512 at request time).

ABO's "small" images are only 256 px, so we fetch each "original" photo by itself from
the public bucket instead of the 118 GB archive. Re-running skips photos already saved.

Run: .venv-serve/bin/python scripts/stage2/build_products.py   (~30-45 min)
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from catalog_cascade import abo

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "abo"
IMAGES = ROOT / "data" / "images"
BUCKET = "https://amazon-berkeley-objects.s3.amazonaws.com/images/original"
MAX_SIDE = 768
MAX_IMAGES = 3


def select() -> list[dict]:
    """De-duplicated products of the chosen types; shoes capped at 3,000 (fixed seed)."""
    files = sorted((RAW / "listings").glob("*.json.gz"))
    kept = abo.dedupe(x for x in abo.iter_listings(files) if abo.product_type(x) in abo.TYPES)
    groups = abo.group_by_type(kept)
    products = []
    for ptype, items in sorted(groups.items()):
        items = [i for i in items if i.get("main_image_id") and abo.english(i.get("item_name"))]
        cap = abo.CAPS.get(ptype)
        if cap and len(items) > cap:
            items = random.Random(0).sample(items, cap)
        for i in items:
            images = [i["main_image_id"], *(i.get("other_image_id") or [])][:MAX_IMAGES]
            products.append({
                "item_id": i["item_id"],
                "product_type": ptype,
                "title": abo.english(i.get("item_name")),
                "brand": abo.brand(i),
                "images": images,
                "seller_values": {f: v for f in abo.FIELDS if (v := abo.english(i.get(f)))},
            })
    return products


def fetch(image_id: str, path: str) -> bool:
    """Download one original photo, shrink it, save it as JPEG. True if it is on disk."""
    dest = IMAGES / f"{image_id}.jpg"
    if dest.exists():
        return True
    try:
        with urllib.request.urlopen(f"{BUCKET}/{path}", timeout=60) as r:
            image = Image.open(io.BytesIO(r.read())).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        image.save(dest, format="JPEG", quality=90)
        return True
    except Exception as exc:  # a missing photo drops that photo, not the product
        print(f"skip {image_id}: {exc}", flush=True)
        return False


def main() -> int:
    products = select()
    print(f"{len(products)} products selected", flush=True)
    with gzip.open(RAW / "images.csv.gz", "rt", encoding="utf-8") as f:
        paths = {row["image_id"]: row["path"] for row in csv.DictReader(f)}
    wanted = sorted({i for p in products for i in p["images"] if i in paths})
    IMAGES.mkdir(parents=True, exist_ok=True)
    print(f"{len(wanted)} photos to fetch", flush=True)
    ok: set[str] = set()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for n, (image_id, saved) in enumerate(
                zip(wanted, pool.map(lambda i: fetch(i, paths[i]), wanted), strict=True), 1):
            if saved:
                ok.add(image_id)
            if n % 500 == 0:
                print(f"{n}/{len(wanted)} photos", flush=True)
    out = []
    for p in products:
        p["images"] = [i for i in p["images"] if i in ok]
        if p["images"]:  # a product with no photo can't be labelled from photos
            out.append(p)
    with open(ROOT / "data" / "products.jsonl", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Done: {len(out)} products, {len(ok)} photos in {IMAGES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
