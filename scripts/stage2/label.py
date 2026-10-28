#!/usr/bin/env python3
"""Label products of some splits with one model served by vLLM. Resumable.

Appends one JSON line per product to --out; products already in --out are skipped, so a
stopped run continues where it left off.

Run (label_all.sh starts the server):
    python scripts/stage2/label.py --model M --splits train,dev --image-size 768 --out F
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from catalog_cascade.labelling import label
from catalog_cascade.smoke import wait_ready

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", required=True, help="comma-separated, e.g. train,dev")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    wanted = set(args.splits.split(","))
    out = Path(args.out)
    done = {json.loads(line)["item_id"] for line in out.open()} if out.exists() else set()
    todo = [p for p in map(json.loads, (ROOT / "data" / "products.jsonl").open())
            if splits.get(p["item_id"]) in wanted and p["item_id"] not in done]
    print(f"{len(todo)} to label ({len(done)} already done)", flush=True)
    if not todo:
        return 0

    ready, waited, why = wait_ready(args.base_url, 2400, args.server_pid)
    if not ready:
        print(f"Server not ready after {waited:.0f} s: {why}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        rows = pool.map(lambda p: label(args.base_url, args.model, p, ROOT / "data" / "images",
                                        args.image_size), todo)
        for n, row in enumerate(rows, 1):
            if "error" in row:
                errors += 1  # not written, so a rerun retries it
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"{n}/{len(todo)} labelled, {errors} errors", flush=True)
    print(f"Done: {len(todo) - errors} labelled, {errors} errors", flush=True)
    return 0 if errors < len(todo) * 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
