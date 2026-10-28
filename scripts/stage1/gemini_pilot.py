#!/usr/bin/env python3
"""Stage 1 API mini-pilot: 20 Gemini 3.1 Pro calls on real ABO products.

It measures what the cost plan only assumed:
- real input, output and thinking tokens per call at low media resolution;
- whether Gemini returns log-probs for image inputs (if not, API answers stay
  "measured, not certified" in the design);
- whether the JSON answers parse, and how often they agree with the seller's values.

Cost: about $0.21 (about Rs 20) at the plan's assumptions. A hard cap (default $0.60)
stops the run early if thinking is much longer than assumed.

Setup (once):  uv venv .venv-api && uv pip install --python .venv-api/bin/python google-genai
Run:           export GEMINI_API_KEY=...   (never commit it; this script never prints it)
               .venv-api/bin/python scripts/stage1/gemini_pilot.py
Needs data/raw/pilot/ from scripts/stage1/abo_check.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-3.1-pro-preview"
PRICE_IN, PRICE_OUT = 2.00, 12.00  # $ per 1M tokens, standard tier; thinking is billed as output
USD_TO_INR = 95.46  # last rate found (25 Aug 2026); an assumption, see the design doc
FIELDS = ("color", "material", "pattern", "style")

PROMPT = (
    "You label product attributes for an online catalogue. Using the photos and the title, "
    "give the product's colour, material, pattern and style. Use short lowercase words, and "
    '"none" when the attribute does not apply or cannot be seen. Product type: {ptype}. '
    "Title: {title}"
)
SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in FIELDS},
    "required": list(FIELDS),
}


def build_config(types, attempt: dict):
    """GenerateContentConfig with whichever optional features this attempt still asks for."""
    kwargs = {"response_mime_type": "application/json", "response_schema": SCHEMA}
    if attempt["thinking_low"]:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    if attempt["media_low"]:
        kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
    if attempt["logprobs"]:
        kwargs["response_logprobs"] = True
        kwargs["logprobs"] = 5
    return types.GenerateContentConfig(**kwargs)


def agreement(answer: dict, seller: dict) -> dict:
    """Per field: does the API's value appear in the seller's value (rough, lowercase)?"""
    out = {}
    for f in FIELDS:
        if f in seller and isinstance(answer.get(f), str):
            a, s = answer[f].strip().lower(), seller[f].strip().lower()
            out[f] = bool(a) and (a in s or s in a)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cap-usd", type=float, default=0.60)
    ap.add_argument("--pilot-dir", default=str(ROOT / "data" / "raw" / "pilot"))
    args = ap.parse_args(argv)

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY first (from aistudio.google.com); nothing was sent.")
        return 2
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Install the SDK: uv venv .venv-api && "
              "uv pip install --python .venv-api/bin/python google-genai")
        return 2

    pilot = Path(args.pilot_dir)
    items = [json.loads(line) for line in (pilot / "items.jsonl").read_text("utf-8").splitlines()
             if line.strip()][: args.n]
    client = genai.Client()

    # Try every feature first; drop one only if the API rejects it, and record which.
    attempt = {"thinking_low": True, "media_low": True, "logprobs": True}
    rejected: dict[str, str] = {}
    rows, spent = [], 0.0
    for item in items:
        if spent >= args.cap_usd:
            print(f"Stopping: spend reached the ${args.cap_usd} cap.")
            break
        parts = [types.Part.from_bytes(data=(pilot / name).read_bytes(), mime_type="image/jpeg")
                 for name in item["images"]]
        parts.append(PROMPT.format(ptype=item["product_type"], title=item["title"]))
        response, error = None, None
        for _ in range(4):
            try:
                start = time.perf_counter()
                response = client.models.generate_content(
                    model=MODEL, contents=parts, config=build_config(types, attempt))
                latency = time.perf_counter() - start
                break
            except Exception as exc:  # find which feature the API refused, drop it, retry
                text = str(exc).lower()
                for feature, hint in (("logprobs", "logprob"), ("thinking_low", "thinking"),
                                      ("media_low", "media")):
                    if attempt[feature] and hint in text:
                        attempt[feature] = False
                        rejected[feature] = str(exc)[:300]
                        break
                else:
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    break
        if response is None:
            rows.append({"item_id": item["item_id"], "error": error})
            continue
        usage = response.usage_metadata
        tokens_in = usage.prompt_token_count or 0
        tokens_out = usage.candidates_token_count or 0
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        cost = (tokens_in * PRICE_IN + (tokens_out + thoughts) * PRICE_OUT) / 1e6
        spent += cost
        candidate = response.candidates[0] if response.candidates else None
        has_logprobs = bool(candidate and getattr(candidate, "logprobs_result", None))
        try:
            answer = json.loads(response.text or "")
        except (ValueError, TypeError):
            answer = None
        rows.append({
            "item_id": item["item_id"],
            "product_type": item["product_type"],
            "images": len(item["images"]),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "thinking_tokens": thoughts,
            "cost_usd": round(cost, 5),
            "latency_s": round(latency, 2),
            "json_ok": isinstance(answer, dict),
            "logprobs_returned": has_logprobs,
            "answer": answer,
            "agrees_with_seller": agreement(answer or {}, item.get("seller_values", {})),
        })
        print(f"{len(rows):2d}. {item['product_type']:<8} in {tokens_in} out {tokens_out} "
              f"thinking {thoughts}  ${cost:.4f}", flush=True)

    ok = [r for r in rows if "error" not in r]
    mean = lambda key: round(sum(r[key] for r in ok) / len(ok), 1) if ok else None  # noqa: E731
    agree = [v for r in ok for v in r["agrees_with_seller"].values()]
    summary = {
        "model": MODEL,
        "calls": len(rows),
        "errors": len(rows) - len(ok),
        "features_rejected": rejected,
        "features_used": attempt,
        "mean_tokens_in": mean("tokens_in"),
        "mean_tokens_out": mean("tokens_out"),
        "mean_thinking_tokens": mean("thinking_tokens"),
        "plan_assumed_thinking_tokens": 500,
        "json_ok_rate": round(sum(r["json_ok"] for r in ok) / len(ok), 3) if ok else None,
        "logprobs_for_images": any(r["logprobs_returned"] for r in ok) if ok else None,
        "seller_agreement_rate": round(sum(agree) / len(agree), 3) if agree else None,
        "spent_usd": round(spent, 4),
        "spent_inr_assumed_rate": round(spent * USD_TO_INR, 1),
        "rows": rows,
    }
    out = ROOT / "reports" / "gemini_pilot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSpent ${summary['spent_usd']} (about Rs {summary['spent_inr_assumed_rate']}); "
          f"mean thinking tokens {summary['mean_thinking_tokens']} (plan assumed 500); "
          f"log-probs for images: {summary['logprobs_for_images']}. "
          f"Seller agreement is rough and is not accuracy: seller values are ~10% wrong.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
