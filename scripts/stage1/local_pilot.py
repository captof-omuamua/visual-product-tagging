#!/usr/bin/env python3
"""Stage 1 local mini-pilot: the 20 real ABO pilot products through a local model (₹0).

It answers the open worry from the smoke test: on real products, is the model's confidence
spread out, or stuck near 100%? (If stuck, the hand-off combiner has no signal.)

Per item it asks for colour, material, pattern and style as JSON whose values must come
from short closed lists (a provisional stand-in for the stage 2 schema), with log-probs on.
A field's confidence is exp(sum of the log-probs of its value tokens).

Usage (scripts/stage1/local_pilot.sh starts and stops the server for you):
    python scripts/stage1/local_pilot.py --model MODEL --image-size 768 --out reports/x.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from catalog_cascade.checks import _tokens, parse, payload
from catalog_cascade.smoke import wait_ready

ROOT = Path(__file__).resolve().parents[2]
# Provisional value lists, only for this pilot; stage 2 builds the real ones.
VALUES = {
    "color": ["black", "white", "grey", "brown", "beige", "blue", "green", "red", "pink",
              "yellow", "orange", "purple", "gold", "silver", "multicolor"],
    "material": ["wood", "metal", "glass", "plastic", "fabric", "leather", "faux leather",
                 "rubber", "wool", "cotton", "jute", "synthetic", "ceramic", "stone"],
    "pattern": ["solid", "striped", "floral", "geometric", "abstract", "checked", "printed"],
    "style": ["modern", "contemporary", "mid-century", "traditional", "rustic", "industrial",
              "casual", "formal", "athletic", "bohemian", "minimalist"],
}
FIELDS = tuple(VALUES)
PROMPT = ("You label product attributes for an online catalogue. Using the photos and the "
          "title, give the product's colour, material, pattern and style. Pick from the "
          "allowed values; use \"other\" if none fits and \"none\" if it does not apply or "
          "cannot be seen. Product type: {ptype}. Title: {title}")
SCHEMA = {
    "type": "object",
    "properties": {f: {"enum": v + ["other", "none"]} for f, v in VALUES.items()},
    "required": list(FIELDS),
    "additionalProperties": False,
}


def image_url(path: Path, size: int) -> str:
    """Shrink so the longer side is at most `size` px, and inline it as a JPEG data URL."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def field_confidence(tokens: list[tuple[str, float]], field: str) -> float | None:
    """exp(sum of log-probs) of the tokens inside this field's value.

    The value is found right after its own key ("color": "...") rather than by searching
    for the word, because two fields often share a value such as "none" or "other".
    """
    text = "".join(t for t, _ in tokens)
    key = text.find(f'"{field}"')
    if key < 0:
        return None
    a = text.find('"', text.find(":", key)) + 1  # first character of the value
    b = text.find('"', a)                       # closing quote
    lp, pos = [], 0
    for token, logprob in tokens:
        if pos < b and pos + len(token) > a:   # token overlaps the value's characters
            lp.append(logprob)
        pos += len(token)
    return round(math.exp(sum(lp)), 4) if lp else None


def label(base_url: str, model: str, item: dict, pilot: Path, size: int) -> dict:
    """One request; returns the answer, per-field confidence and latency."""
    urls = [image_url(pilot / name, size) for name in item["images"][:3]]
    prompt = PROMPT.format(ptype=item["product_type"], title=item["title"])
    body = payload(model, prompt, urls, SCHEMA, logprobs=True, max_tokens=96)
    start = time.perf_counter()
    raw = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600).json()
    latency = time.perf_counter() - start
    answer = parse(raw["choices"][0]["message"]["content"])
    tokens = _tokens(raw["choices"][0])
    confidence = {f: field_confidence(tokens, f) for f in FIELDS} if answer else {}
    return {"item_id": item["item_id"], "product_type": item["product_type"],
            "answer": answer, "confidence": confidence, "latency_s": round(latency, 2),
            "seller_values": item.get("seller_values", {})}


def seller_agrees(value: str, seller: str) -> bool:
    """Rough: does the chosen word appear in the seller's free text (or vice versa)?"""
    a, s = value.lower(), seller.lower()
    return a not in ("none", "other") and (a in s or s in a)


def summarize(rows: list[dict]) -> dict:
    conf = [c for r in rows for c in r["confidence"].values() if c is not None]
    agree = [seller_agrees(r["answer"][f], r["seller_values"][f])
             for r in rows if r["answer"] for f in FIELDS if f in r["seller_values"]]
    q = statistics.quantiles(conf, n=10) if len(conf) >= 2 else []
    return {
        "items": len(rows),
        "json_ok_rate": round(sum(r["answer"] is not None for r in rows) / len(rows), 3),
        "fields_scored": len(conf),
        "share_conf_over_0.99": round(sum(c > 0.99 for c in conf) / len(conf), 3),
        "share_conf_under_0.80": round(sum(c < 0.80 for c in conf) / len(conf), 3),
        "conf_deciles": [round(x, 3) for x in q],
        "conf_min": min(conf),
        "seller_agreement_rate": round(sum(agree) / len(agree), 3) if agree else None,
        "seller_fields_compared": len(agree),
        "p50_latency_s": statistics.median(r["latency_s"] for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--pilot-dir", default=str(ROOT / "data" / "raw" / "pilot"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ready, waited, why = wait_ready(args.base_url, 2400, args.server_pid)
    if not ready:
        print(f"Server not ready after {waited:.0f} s: {why}")
        return 1
    pilot = Path(args.pilot_dir)
    items = [json.loads(line) for line in (pilot / "items.jsonl").read_text("utf-8").splitlines()
             if line.strip()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda it: label(args.base_url, args.model, it, pilot,
                                              args.image_size), items))
    for r in rows:
        print(r["product_type"], r["answer"], r["confidence"], flush=True)
    report = {"model": args.model, "image_size": args.image_size,
              "summary": summarize(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
