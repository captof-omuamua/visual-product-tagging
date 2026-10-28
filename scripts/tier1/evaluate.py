#!/usr/bin/env python3
"""Score models on the test split against the provisional answer key (Qwen3-VL-8B).

Per model and field:
- agree: share of test products where the model's value equals the answer key's.
- seller: share equal to the seller's value, where the seller's text maps to a list value
  (a second, independent check; seller values are ~10% wrong).
- cover@95: the largest share of products the model could answer, taking its most
  confident first, while still agreeing with the key at least 95% of the time.
  This is a preview of the cascade's coverage, not a certificate.

All numbers are provisional: the key is another model, not checked by hand yet.
Run: .venv-serve/bin/python scripts/tier1/evaluate.py   -> reports/tier1_v0_eval.{json,md}
"""

from __future__ import annotations

import json
from pathlib import Path

from catalog_cascade.schema import fields_for, normalize

ROOT = Path(__file__).resolve().parents[2]
KEY = "qwen3vl8b"
MODELS = {"Tier 1 v0 (Qwen3.5-4B, fine-tuned)": "tier1_v0",
          "Qwen3.5-4B, not fine-tuned": "qwen35_4b_base",
          "Gemma 4 26B (Tier 2)": "gemma4"}
FIELDS = ("color", "material", "pattern", "style", "shape", "finish")


def load(name: str) -> dict[str, dict]:
    path = ROOT / "data" / "labels" / f"{name}.jsonl"
    return {r["item_id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def cover_at(pairs: list[tuple[float, bool]], target: float = 0.95) -> float:
    """Largest share answerable, most confident first, keeping agreement >= target."""
    pairs = sorted(pairs, key=lambda x: -x[0])
    best, right = 0, 0
    for n, (_, ok) in enumerate(pairs, 1):
        right += ok
        if right / n >= target:
            best = n
    return best / len(pairs) if pairs else 0.0


def main() -> int:
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    products = {p["item_id"]: p for p in map(json.loads, (ROOT / "data" / "products.jsonl").open())
                if splits.get(p["item_id"]) == "test"}
    key = load(KEY)
    results: dict = {}
    for label, name in MODELS.items():
        preds = load(name)
        if not preds:
            continue
        results[label] = {}
        for field in FIELDS:
            pairs, seller = [], []
            for item_id, p in products.items():
                if field not in fields_for(p["product_type"]):
                    continue
                pred = (preds.get(item_id) or {}).get("answer") or {}
                truth = (key.get(item_id) or {}).get("answer") or {}
                if field not in pred or field not in truth:
                    continue
                conf = (preds[item_id].get("confidence") or {}).get(field) or 0.0
                pairs.append((conf, pred[field] == truth[field]))
                s = normalize(field, p["product_type"], p["seller_values"].get(
                    {"shape": "item_shape", "finish": "finish_type"}.get(field, field)))
                if s:
                    seller.append(pred[field] == s)
            if pairs:
                results[label][field] = {
                    "n": len(pairs),
                    "agree": round(sum(ok for _, ok in pairs) / len(pairs), 3),
                    "seller": round(sum(seller) / len(seller), 3) if seller else None,
                    "seller_n": len(seller),
                    "cover@95": round(cover_at(pairs), 3),
                }
    (ROOT / "reports" / "tier1_v0_eval.json").write_text(json.dumps(results, indent=2))

    lines = ["# Tier 1 v0: test split, provisional", "",
             f"Answer key: {KEY} (outside the cascade). Agree = same value as the key; "
             "seller = same as the seller's mapped value; cover@95 = share answerable at "
             "95% agreement, most confident first.", ""]
    for field in FIELDS:
        rows = [(m, r[field]) for m, r in results.items() if field in r]
        if not rows:
            continue
        lines += [f"## {field} (n = {rows[0][1]['n']})", "",
                  "| Model | Agree | Seller | Cover@95 |", "|---|---|---|---|"]
        for m, r in rows:
            seller = f"{r['seller']:.0%} (n={r['seller_n']})" if r["seller"] is not None else "-"
            lines.append(f"| {m} | {r['agree']:.0%} | {seller} | {r['cover@95']:.0%} |")
        lines.append("")
    (ROOT / "reports" / "tier1_v0_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
