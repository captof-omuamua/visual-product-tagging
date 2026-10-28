#!/usr/bin/env bash
# Stage 1 in one command. After a failure, later steps still run where they can, so the
# report shows everything. First run: 1-2 hours, mostly downloads (about 60 GB).
#   SKIP_TIER2=1        skip the two large models (saves ~40 GB and most of the time)
#   GEMINI_API_KEY=...  runs the 20-call Gemini mini-pilot (about Rs 20); skipped if unset
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p reports
rm -f reports/train_smoke.json reports/serve_runtime.json reports/serve_merged.json \
  reports/tier2_*.json reports/abo_check.json reports/gemini_pilot.json \
  reports/stage1_report.md reports/stage1_report.json

echo "== 1/7 Environment check =="
python3 scripts/stage1/check_env.py || exit 1

echo "== 2/7 Python environments (uv) =="
if ! bash scripts/stage1/setup_envs.sh 2>&1 | tee reports/setup.log; then
  echo "Setup failed: see reports/setup.log"
  python3 scripts/stage1/report.py
  exit 1
fi

echo "== 3/7 ABO data check (90 MB download) =="
.venv-train/bin/python scripts/stage1/abo_check.py 2>&1 | tee reports/abo_check.log

echo "== 4/7 Training smoke test (Qwen3.5-4B, 16-bit LoRA, merge; about 10 minutes) =="
.venv-train/bin/python scripts/stage1/train_smoke.py 2>&1 | tee reports/train_smoke.log

echo "== 5/7 Serving smoke test (run-time adapter, then merged weights) =="
bash scripts/stage1/serve_smoke.sh 2>&1 | tee reports/serve_smoke.log

echo "== 6/7 Large models (Qwen3.5-27B and Gemma 4 26B-A4B) =="
if [ "${SKIP_TIER2:-0}" = "1" ]; then
  echo "Skipped (SKIP_TIER2=1)."
else
  bash scripts/stage1/tier2_smoke.sh 2>&1 | tee reports/tier2_smoke.log
fi

echo "== 7/7 Gemini mini-pilot =="
if [ -n "${GEMINI_API_KEY:-}" ]; then
  .venv-api/bin/python scripts/stage1/gemini_pilot.py 2>&1 | tee reports/gemini_pilot.log
else
  echo "Skipped: GEMINI_API_KEY is not set (run it later on its own; see docs/stage1.md)."
fi

echo "== Report =="
python3 scripts/stage1/report.py
