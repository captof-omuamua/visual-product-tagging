"""Stage 1 serving smoke test, run against a live vLLM server.

Checks, in order:
1. The server becomes healthy, and lists the base model (and the adapter, if trained).
2. Burst: 200 requests to the base model at once. Every one must succeed.
3. Sustained: 200 requests to the LoRA adapter at concurrency 50.
4. After the load, /health still answers and a fresh request succeeds. This
   catches the deadlock one RTX 5090 user reported on vLLM 0.11.2 in June 2026.
5. LoRA effect: the adapter's exact-format rate against the base model's. This
   is a warning, not a failure, because 30 training steps make a smoke test, not
   a model. It exists to catch an adapter that loads but is silently ignored.
6. With --grammar: schema enforcement and log-probs (catalog_cascade.checks).
7. With --answers-out / --compare-with: greedy answers saved, or compared with the
   answers from the other serving (run-time adapter against merged weights).

Every request carries --images-per-item images of --image-size px, like a real listing.

Usage (scripts/stage1/serve_smoke.sh does this for you):
    python -m catalog_cascade.smoke --model Qwen/Qwen3.5-4B --adapter smoke-lora
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import requests

from . import checks
from .loadtest import run_load, send_one, summarize
from .synthetic import EVAL_SEED, PROMPT, Item, data_url, make_items, views

JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
LORA_EFFECT_MARGIN = 0.2


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_ready(
    base_url: str, timeout_s: float, server_pid: int | None = None, poll_s: float = 5.0
) -> tuple[bool, float, str]:
    """Poll /health until it answers 200, the server process dies, or time runs out."""
    start = time.monotonic()
    while True:
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True, time.monotonic() - start, "ready"
        except requests.RequestException:
            pass
        if server_pid is not None and not _alive(server_pid):
            return False, time.monotonic() - start, "server_exited"
        if time.monotonic() - start > timeout_s:
            return False, time.monotonic() - start, "startup_timeout"
        time.sleep(poll_s)


def healthy(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=10).status_code == 200
    except requests.RequestException:
        return False


def list_models(base_url: str) -> list[str]:
    try:
        body = requests.get(f"{base_url}/v1/models", timeout=30).json()
        return [m["id"] for m in body.get("data", [])]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return []


def server_version(base_url: str) -> str | None:
    try:
        return requests.get(f"{base_url}/version", timeout=10).json().get("version")
    except (requests.RequestException, ValueError, AttributeError):
        return None


def parse_attributes(text: str | None) -> dict[str, str] | None:
    """First flat JSON object in the text, keys and values lower-cased; None if none parses."""
    if not text:
        return None
    for match in JSON_OBJECT.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return {str(k).strip().lower(): str(v).strip().lower() for k, v in obj.items()}
    return None


def score(outputs: Sequence[str | None], items: Sequence[Item]) -> dict:
    """Exact-format rate, JSON rate and per-field accuracy against the drawn labels."""
    n = len(items)
    if n == 0:
        return {"n": 0}
    exact = json_ok = color_ok = shape_ok = 0
    for text, item in zip(outputs, items, strict=True):
        if text is not None and text.strip() == item.target:
            exact += 1
        parsed = parse_attributes(text)
        if parsed is None:
            continue
        json_ok += 1
        if (parsed.get("color") or parsed.get("colour")) == item.color:
            color_ok += 1
        if parsed.get("shape") == item.shape:
            shape_ok += 1
    return {
        "n": n,
        "exact_format_rate": round(exact / n, 3),
        "json_rate": round(json_ok / n, 3),
        "color_accuracy": round(color_ok / n, 3),
        "shape_accuracy": round(shape_ok / n, 3),
    }


def tail(path: str | None, lines: int = 80) -> list[str]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()[-lines:]


def _outputs(
    base_url: str, model: str, url_sets: Sequence[Sequence[str]], timeout_s: float
) -> list[str | None]:
    payloads = [checks.payload(model, PROMPT, urls) for urls in url_sets]
    results, _ = run_load(base_url, payloads, concurrency=8, timeout_s=timeout_s)
    return [r.text if r.ok else None for r in results]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="Base model id, as served")
    ap.add_argument("--adapter", default=None, help="LoRA module name, if one was trained")
    ap.add_argument("--burst", type=int, default=200)
    ap.add_argument("--sustained", type=int, default=200)
    ap.add_argument("--sustained-concurrency", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=300, help="Per-request read timeout (s)")
    ap.add_argument("--startup-timeout", type=float, default=1200)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--images-per-item", type=int, default=3)
    ap.add_argument("--grammar", action="store_true", help="run the schema and log-prob checks")
    ap.add_argument("--answers-out", default=None, help="save greedy answers for comparison")
    ap.add_argument("--compare-with", default=None, help="answers saved by the other serving")
    ap.add_argument("--out", default="reports/serve_smoke.json")
    args = ap.parse_args(argv)
    base = args.base_url.rstrip("/")

    report: dict = {"model": args.model, "adapter": args.adapter, "checks": {}, "ok": False}
    results: dict[str, bool] = report["checks"]

    ready, startup_s, why = wait_ready(base, args.startup_timeout, args.server_pid)
    report["startup"] = why
    report["startup_s"] = round(startup_s, 1)
    results["server_ready"] = ready
    if not ready:
        report["server_log_tail"] = tail(args.server_log)
        _write(report, args.out)
        print(f"FAIL: server not ready ({why}); last log lines are in {args.out}")
        return 1

    report["vllm_version"] = server_version(base)
    report["models"] = models = list_models(base)
    results["base_listed"] = args.model in models
    if args.adapter:
        results["adapter_listed"] = args.adapter in models

    items = make_items(n_per_combo=2, seed=EVAL_SEED, size=args.image_size)  # 48, never seen
    urls = [
        [data_url(im) for im in views(it, args.images_per_item, EVAL_SEED, args.image_size)]
        for it in items
    ]

    print(f"Burst: {args.burst} requests at once to {args.model} ...", flush=True)
    burst = [checks.payload(args.model, PROMPT, urls[i % len(urls)]) for i in range(args.burst)]
    load, wall = run_load(base, burst, concurrency=args.burst, timeout_s=args.timeout)
    report["burst"] = {"concurrency": args.burst, **summarize(load, wall)}
    results["burst_all_ok"] = report["burst"]["failed"] == 0

    if args.adapter and results.get("adapter_listed"):
        print(f"Sustained: {args.sustained} requests to {args.adapter} ...", flush=True)
        sustained = [
            checks.payload(args.adapter, PROMPT, urls[i % len(urls)])
            for i in range(args.sustained)
        ]
        load, wall = run_load(
            base, sustained, concurrency=args.sustained_concurrency, timeout_s=args.timeout
        )
        report["sustained"] = {
            "concurrency": args.sustained_concurrency,
            **summarize(load, wall),
        }
        results["sustained_all_ok"] = report["sustained"]["failed"] == 0

    fresh = send_one(base, checks.payload(args.model, PROMPT, urls[0]), timeout_s=120)
    results["healthy_after_load"] = healthy(base) and fresh.ok

    eval_items = items[: args.eval_n]
    eval_urls = urls[: args.eval_n]
    base_out = _outputs(base, args.model, eval_urls, args.timeout)
    report["base_quality"] = score(base_out, eval_items)
    adapter_out: list[str | None] = [None] * len(eval_items)
    if args.adapter and results.get("adapter_listed"):
        adapter_out = _outputs(base, args.adapter, eval_urls, args.timeout)
        report["adapter_quality"] = score(adapter_out, eval_items)
        gain = (
            report["adapter_quality"]["exact_format_rate"]
            - report["base_quality"]["exact_format_rate"]
        )
        report["lora_effect_visible"] = gain >= LORA_EFFECT_MARGIN
    report["examples"] = [
        {"target": it.target, "base": b, "adapter": a}
        for it, b, a in list(zip(eval_items, base_out, adapter_out, strict=True))[:4]
    ]

    if args.grammar:
        print("Grammar and log-prob checks ...", flush=True)
        report["grammar"] = checks.grammar_check(base, args.model, PROMPT, eval_urls,
                                                 args.timeout)
        results["grammar_enforced"] = report["grammar"]["ok"]

    # Greedy answers from whichever model answers in production in this phase.
    answering = args.adapter if args.adapter and results.get("adapter_listed") else args.model
    if args.answers_out or args.compare_with:
        mine = checks.answers(base, answering, PROMPT, eval_urls, args.timeout)
        if args.answers_out:
            Path(args.answers_out).write_text(json.dumps(mine, indent=2), encoding="utf-8")
        if args.compare_with and Path(args.compare_with).exists():
            other = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
            report["merged_vs_runtime_match"] = checks.match_rate(mine, other)

    report["ok"] = all(results.values())
    _write(report, args.out)
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(f"FAIL: serving smoke test (failed: {', '.join(failed)})")
    else:
        print("PASS: serving smoke test")
    return 0 if report["ok"] else 1


def _write(report: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
