"""Stage 1 check of a local large model (Tier 2 or a teacher), against a live vLLM server.

Questions from the design doc's "Still uncertain" list (item 4):
- Does it load in 24 GB at 4k context with 3 images per request?
- How fast is it on image requests? (Teachers label ~8,600 items overnight.)
- Is the output schema enforced? Gemma 4 once skipped enforcement silently (vLLM #39130),
  so the forced-schema check from catalog_cascade.checks runs here too.
- Did vLLM fall back to the slow Triton attention path? (Gemma 4, vLLM #38887)

Usage (scripts/stage1/tier2_smoke.sh does this for you):
    python -m catalog_cascade.tier2 --model Qwen/Qwen3.5-27B-GPTQ-Int4 --server-log LOG
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from . import checks
from .loadtest import run_load, summarize
from .smoke import tail, wait_ready
from .synthetic import EVAL_SEED, PROMPT, data_url, make_items, views

# Log lines worth surfacing. Matching is case-insensitive.
LOG_FLAGS = {
    "triton_attention_fallback": r"TRITON_ATTN",
    "out_of_memory": r"out of memory|CUDA OOM|OutOfMemoryError",
    "mamba_cache_error": r"mamba.*cache|cudagraph.*capture",
}


def scan_log(lines: Sequence[str]) -> dict[str, bool]:
    text = "\n".join(lines)
    return {
        name: bool(re.search(pattern, text, re.IGNORECASE)) for name, pattern in LOG_FLAGS.items()
    }


def overnight_hours(requests_per_s: float | None, items: int = 8600) -> float | None:
    """Hours to label `items` products once at the measured rate (inf: assumes it holds)."""
    if not requests_per_s:
        return None
    return round(items / requests_per_s / 3600, 1)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--images-per-item", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    base = args.base_url.rstrip("/")

    report: dict = {"model": args.model, "ok": False}
    ready, startup_s, why = wait_ready(base, args.startup_timeout, args.server_pid)
    report.update(startup=why, startup_s=round(startup_s, 1))
    if not ready:
        log = tail(args.server_log, 120)
        report.update(server_log_tail=log, log_flags=scan_log(log))
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"FAIL: {args.model} did not start ({why}); see {args.out}")
        return 1

    items = make_items(n_per_combo=3, seed=EVAL_SEED, size=args.image_size)[: args.n]
    urls = [
        [data_url(im) for im in views(it, args.images_per_item, EVAL_SEED, args.image_size)]
        for it in items
    ]
    enum_schema = checks.schema(checks.COLORS, checks.SHAPES)
    bodies = [checks.payload(args.model, PROMPT, u, enum_schema) for u in urls]
    print(f"{args.n} image requests to {args.model} at concurrency {args.concurrency} ...",
          flush=True)
    results, wall = run_load(base, bodies, args.concurrency, args.timeout)
    load = summarize(results, wall)
    answers = [checks.parse(r.text) if r.ok else None for r in results]
    right = sum(
        1 for a, it in zip(answers, items, strict=True)
        if a and a.get("color") == it.color and a.get("shape") == it.shape
    )
    report["load"] = load
    report["synthetic_accuracy"] = round(right / len(items), 3) if items else None
    report["overnight_hours_for_8600_items"] = overnight_hours(load["requests_per_s"])

    print("Grammar check ...", flush=True)
    report["grammar"] = checks.grammar_check(base, args.model, PROMPT, urls[:10], args.timeout)
    report["log_flags"] = scan_log(tail(args.server_log, 4000))
    report["ok"] = load["failed"] == 0 and report["grammar"]["forced_enforced_rate"] == 1.0
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{'PASS' if report['ok'] else 'FAIL'}: {args.model}: {load['ok']}/{load['requests']} ok, "
        f"{load['requests_per_s']} req/s (about {report['overnight_hours_for_8600_items']} h for "
        f"8,600 items), schema enforced {report['grammar']['forced_enforced_rate']}, "
        f"flags {[k for k, v in report['log_flags'].items() if v] or 'none'}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
