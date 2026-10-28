#!/usr/bin/env bash
# Stage 1 serving smoke test, in two phases (design doc: merge the LoRA before serving).
#   A. Base Qwen3.5-4B plus the adapter loaded at run time: burst, sustained load on the
#      adapter, health afterwards; saves the adapter's greedy answers.
#   B. The merged 16-bit model, which is how Tier 1 is served: burst, health, the grammar
#      and log-prob checks, and the share of answers identical to phase A.
# Both phases: thinking off server-wide, no reasoning parser, 3 images per request.
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here (no nvcc); see docs/stage1.md

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
# 0.85 of 24 GB leaves headroom for the desktop; if vLLM reports too little free memory,
# rerun with GPU_UTIL=0.75. The rest of the 200-request burst waits in vLLM's queue,
# which is the path the test exercises.
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ADAPTER_DIR="outputs/smoke-lora"
MERGED_DIR="outputs/smoke-merged"
mkdir -p reports

common=(--host 127.0.0.1 --port "$PORT" --max-model-len 8192
  --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_NUM_SEQS"
  --limit-mm-per-prompt '{"image": 3}'
  --default-chat-template-kwargs '{"enable_thinking": false}')

run_phase() {  # $1 phase name, then: vllm serve args ... -- smoke args ...
  local phase="$1"; shift
  local serve_args=() smoke_args=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do serve_args+=("$1"); shift; done
  shift
  smoke_args=("$@")
  local log="reports/vllm_${phase}.log"
  echo "== Phase ${phase}: .venv-serve/bin/vllm serve ${serve_args[*]} =="
  echo "Server log: $log (the first start compiles kernels; allow a few minutes)"
  .venv-serve/bin/vllm serve "${serve_args[@]}" "${common[@]}" > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python -m catalog_cascade.smoke --base-url "http://127.0.0.1:$PORT" \
    --server-pid "$pid" --server-log "$log" "${smoke_args[@]}"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5  # let the GPU memory free before the next phase
  return $status
}

rm -f reports/answers_runtime.json
if [ -f "$ADAPTER_DIR/adapter_config.json" ]; then
  rank=$(.venv-serve/bin/python -c "import json; print(json.load(open('$ADAPTER_DIR/adapter_config.json'))['r'])")
  run_phase runtime "$MODEL" --enable-lora --max-lora-rank "$rank" \
    --lora-modules "smoke-lora=$ADAPTER_DIR" \
    -- --model "$MODEL" --adapter smoke-lora --answers-out reports/answers_runtime.json \
    --out reports/serve_runtime.json
  status_a=$?
else
  echo "No adapter in $ADAPTER_DIR (training failed or was skipped): testing the base model only."
  run_phase runtime "$MODEL" -- --model "$MODEL" --grammar --out reports/serve_runtime.json
  exit $?
fi

if [ -f "$MERGED_DIR/config.json" ]; then
  run_phase merged "$MERGED_DIR" -- --model "$MERGED_DIR" --grammar \
    --compare-with reports/answers_runtime.json --out reports/serve_merged.json
  status_b=$?
else
  echo "No merged model in $MERGED_DIR: the merge step failed; see reports/train_smoke.json."
  status_b=1
fi
[ "$status_a" -eq 0 ] && [ "$status_b" -eq 0 ]
