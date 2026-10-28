#!/usr/bin/env python3
"""Merge the stage 1 JSON reports into reports/stage1_report.md (standard library only).

It answers the first seven "Still uncertain" items in the design doc, one row each.
Exit code 0 means PASS. Warnings do not fail the run but are listed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORY_LIMIT_GB = 22.0


def load(reports: Path, name: str) -> dict | None:
    path = reports / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "error": f"{name} is not valid JSON"}


def load_all(reports: Path) -> dict:
    found = {
        "env": load(reports, "env_check.json"),
        "abo": load(reports, "abo_check.json"),
        "train": load(reports, "train_smoke.json"),
        "runtime": load(reports, "serve_runtime.json"),
        "merged": load(reports, "serve_merged.json"),
        "pilot": load(reports, "gemini_pilot.json"),
    }
    found["tier2"] = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(reports.glob("tier2_*.json"))
    ]
    return found


def _failed_checks(serve: dict, label: str) -> list[str]:
    return [f"{label}: check '{name}' failed." for name, ok in
            (serve.get("checks") or {}).items() if not ok]


def verdict(r: dict) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    env, abo, train = r["env"], r["abo"], r["train"]
    runtime, merged, tier2, pilot = r["runtime"], r["merged"], r["tier2"], r["pilot"]

    if env is None:
        failures.append("The environment check did not run.")
    elif not env.get("ok"):
        failures.extend(env.get("blockers") or ["The environment check failed."])
    else:
        warnings.extend(env.get("warnings") or [])

    if train is None:
        failures.append("The training smoke test did not run.")
    elif not train.get("ok"):
        failures.append(f"Training failed at '{train.get('failed_stage', '?')}': "
                        f"{train.get('error', 'unknown error')}")
    else:
        if not train.get("memory_ok", True):
            warnings.append(f"Training peaked at {train.get('peak_vram_gb')} GB, over the "
                            f"{MEMORY_LIMIT_GB} GB limit: drop to 448 px or two images.")
        if not (train.get("thinking") or {}).get("training_matches_thinking_off", True):
            warnings.append("Training text is not rendered in the thinking-off format; stage 5 "
                            "must render it with enable_thinking=False.")

    if runtime is None:
        failures.append("Serving with the run-time adapter did not run.")
    else:
        failures += _failed_checks(runtime, "Run-time adapter")
    if merged is None:
        failures.append("Serving the merged model did not run.")
    else:
        failures += _failed_checks(merged, "Merged model")
        grammar = merged.get("grammar") or {}
        if grammar and not grammar.get("structure_tokens_near_zero"):
            warnings.append("Structural tokens don't score about 0 under the grammar; the "
                            "confidence signal must be re-derived before stage 6.")
        match = merged.get("merged_vs_runtime_match")
        if match is not None and match < 0.9:
            warnings.append(f"Merged and run-time answers match only {match:.0%}; serve merged "
                            "(already the plan) and note it in the decision record.")

    if not tier2:
        warnings.append("The large-model check did not run (SKIP_TIER2=1?).")
    elif not any(t.get("ok") for t in tier2):
        failures.append("Neither local large model passed; Tier 2 would need the API.")
    else:
        for t in tier2:
            if not t.get("ok"):
                warnings.append(f"{t.get('model')} failed its check; the other one carries on.")
            if (t.get("log_flags") or {}).get("triton_attention_fallback"):
                warnings.append(f"{t.get('model')} used the slow Triton attention path.")

    if abo is None:
        warnings.append("The ABO data check did not run.")
    elif abo.get("failing_types"):
        warnings.append("ABO types failing a check: " + ", ".join(abo["failing_types"]) +
                        ". Choose swaps from options B or C before stage 2.")
    if pilot is None:
        warnings.append("The Gemini mini-pilot did not run (set GEMINI_API_KEY).")
    return ("PASS" if not failures else "FAIL"), failures, warnings


def _load_line(block: dict | None) -> str:
    if not block:
        return "not run"
    lat = block.get("latency_s") or {}
    return (f"{block.get('ok')}/{block.get('requests')} ok; p50 {lat.get('p50')} s, "
            f"p95 {lat.get('p95')} s")


def answers(r: dict) -> list[tuple[str, str]]:
    """One line per "Still uncertain" item 1-7."""
    train, runtime, merged = r["train"] or {}, r["runtime"] or {}, r["merged"] or {}
    grammar = merged.get("grammar") or {}
    abo, pilot = r["abo"] or {}, r["pilot"] or {}
    rows = []
    rows.append((
        "1. Qwen3.5-4B trains, merges and serves",
        (f"trained {train.get('steps')} steps, loss {train.get('loss_first')} -> "
         f"{train.get('loss_last')}; merged {train.get('merged_gb')} GB; burst "
         f"{_load_line(merged.get('burst'))}; healthy after load: "
         f"{(merged.get('checks') or {}).get('healthy_after_load')}; merged vs run-time "
         f"answers identical: {merged.get('merged_vs_runtime_match')}")
        if train.get("ok") else f"failed at '{train.get('failed_stage')}'",
    ))
    rows.append((
        "2. Grammar enforced; log-probs as expected",
        (f"forced schema obeyed {grammar.get('forced_enforced_rate')}; enum answers valid "
         f"{grammar.get('enum_valid_rate')}; structural tokens median log-prob "
         f"{(grammar.get('structure_logprob') or {}).get('median')}, value tokens "
         f"{(grammar.get('value_logprob') or {}).get('median')}") if grammar else "not run",
    ))
    rows.append((
        "3. Memory with 3 images of 512 px",
        f"training peak {train.get('peak_vram_gb')} GB (limit {MEMORY_LIMIT_GB}); serving "
        f"burst {_load_line(runtime.get('burst'))}" if train else "not run",
    ))
    tier = "; ".join(
        f"{t.get('model')}: {'ok' if t.get('ok') else 'FAILED'}, "
        f"{(t.get('load') or {}).get('requests_per_s')} req/s (~"
        f"{t.get('overnight_hours_for_8600_items')} h for 8,600 items), schema "
        f"{(t.get('grammar') or {}).get('forced_enforced_rate')}"
        for t in r["tier2"]) or "not run"
    rows.append(("4. Local large models fit and are fast enough", tier))
    if abo:
        types = ", ".join(f"{t['product_type']} {t['products']}"
                          f"{'' if all(t['checks'].values()) else ' (fails)'}"
                          for t in abo.get("types", []))
        reach = abo.get("certificate_reach") or {}
        rows.append(("5. ABO fill rates, families, duplicates, titles", types))
        rows.append((
            "6. Each field can reach its accepted-answer count",
            f"{reach.get('accepted_per_field')} accepted per field (assumed); needs " +
            ", ".join(f"{k}: {v['needed']} ({'ok' if v['reachable'] else 'no'})"
                      for k, v in (reach.get("needed") or {}).items()),
        ))
    else:
        rows += [("5. ABO fill rates, families, duplicates, titles", "not run"),
                 ("6. Each field can reach its accepted-answer count", "not run")]
    rows.append((
        "7. Gemini tokens, thinking and image log-probs",
        (f"{pilot.get('calls')} calls, mean thinking {pilot.get('mean_thinking_tokens')} tokens "
         f"(plan assumed 500); log-probs for images: {pilot.get('logprobs_for_images')}; "
         f"spent ${pilot.get('spent_usd')}") if pilot else "not run",
    ))
    return rows


def render(r: dict, status: str, failures: list[str], warnings: list[str]) -> str:
    env = r["env"] or {}
    gpu = (env.get("gpus") or [{}])[0]
    clean = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = [
        f"# Stage 1 report: {status}",
        "",
        # timezone.utc, not datetime.UTC: this runs on the system Python, which may be 3.10
        f"Run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} on "  # noqa: UP017
        f"{gpu.get('name')} ({gpu.get('vram_gb')} GB, driver {gpu.get('driver')}).",
        "",
        "| Still uncertain (design doc) | What stage 1 found |",
        "|---|---|",
        *[f"| {clean(q)} | {clean(a)} |" for q, a in answers(r)],
    ]
    if failures:
        lines += ["", "## Failures", "", *[f"- {f}" for f in failures]]
    if warnings:
        lines += ["", "## Warnings", "", *[f"- {w}" for w in warnings]]
    lines += ["", "## Next step", ""]
    if status == "PASS":
        lines.append("Commit reports/ and requirements/*-freeze.txt (`git add reports "
                     "requirements && git commit -s`), then send this file to Claude to plan "
                     "stage 2.")
    else:
        lines.append("Send this file and the failing step's log (reports/*.log) to Claude. "
                     "The fallbacks are in docs/stage1.md.")
    return "\n".join(lines) + "\n"


def main(reports_dir: str | None = None) -> int:
    reports = Path(reports_dir) if reports_dir else ROOT / "reports"
    found = load_all(reports)
    status, failures, warnings = verdict(found)
    markdown = render(found, status, failures, warnings)
    (reports / "stage1_report.md").write_text(markdown, encoding="utf-8")
    summary = {"status": status, "failures": failures, "warnings": warnings}
    (reports / "stage1_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(markdown)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
