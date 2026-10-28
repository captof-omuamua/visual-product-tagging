#!/usr/bin/env python3
"""Ask one vLLM-served model every ImplicitAVE test question; save answer, confidence, time.

Confidence = the model's probability for the value it picked (exp of the summed log-probs of
the value's tokens; the JSON around it is forced by the schema, so it carries no signal).
Also records the wall-clock time of the whole run, for throughput and cost.

Run (run_all.sh starts the server):
    python scripts/implicitave/predict.py --model M --image-size 512 --out F
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from catalog_cascade import implicitave as ia
from catalog_cascade.checks import _tokens, parse, payload
from catalog_cascade.labelling import field_confidence, image_url
from catalog_cascade.smoke import wait_ready


def ask(base_url: str, model: str, row: dict, size: int) -> dict:
    body = payload(model, ia.prompt(row), [image_url(Path(row["image"]), size)],
                   ia.schema(row), logprobs=True, max_tokens=32)
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600).json()
        choice = r["choices"][0]
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}
    answer = parse(choice["message"]["content"]) or {}
    return {"id": row["id"], "pred": answer.get("value"),
            "conf": field_confidence(_tokens(choice), "value")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--split", default="test", help="test, or val (to pick the threshold)")
    ap.add_argument("--limit", type=int, default=0, help="random sample of this many rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = ia.load(args.split)
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    ready, waited, why = wait_ready(args.base_url, 2400, args.server_pid)
    if not ready:
        print(f"Server not ready after {waited:.0f} s: {why}")
        return 1
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        preds = list(pool.map(lambda r: ask(args.base_url, args.model, r, args.image_size),
                              rows))
    seconds = time.perf_counter() - start
    errors = sum("error" in p for p in preds)
    out = {"model": args.model, "image_size": args.image_size, "items": len(rows),
           "seconds": round(seconds, 1), "errors": errors, "preds": preds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    right = sum(p.get("pred") == r["value"] for p, r in zip(preds, rows, strict=True))
    print(f"{args.model}: {right}/{len(rows)} right ({right / len(rows):.1%}), "
          f"{errors} errors, {seconds:.0f} s ({len(rows) / seconds:.2f} items/s)", flush=True)
    return 0 if errors < len(rows) * 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
