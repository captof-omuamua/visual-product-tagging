#!/usr/bin/env bash
# Stage 1: can each local large model load and label fast enough on 24 GB?
#   Qwen3.5-27B, GPTQ Int4 (Tier 2 and teacher 1)
#   Gemma 4 26B-A4B, AWQ 4-bit (teacher 2)
# Each runs alone (they can't share the GPU), with thinking off, no reasoning parser,
# 4k context and 3 images of 768 px per request. Each download is about 17-20 GB.
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here (no nvcc); see docs/stage1.md

QWEN="${QWEN:-Qwen/Qwen3.5-27B-GPTQ-Int4}"          # Apache-2.0, from Qwen's own repo
GEMMA="${GEMMA:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}"  # Apache-2.0, a community AWQ build
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.92}"
mkdir -p reports

slug() { echo "$1" | tr '/:' '__'; }

run_model() {  # $1 model id, then extra vllm args
  local model="$1"; shift
  local name; name=$(slug "$model")
  local log="reports/vllm_tier2_${name}.log"
  echo "== ${model} =="
  echo "Server log: $log (first start downloads the weights; allow 10-30 minutes)"
  .venv-serve/bin/vllm serve "$model" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 4096 --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image": 3}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' "$@" > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python -m catalog_cascade.tier2 --model "$model" \
    --base-url "http://127.0.0.1:$PORT" --server-pid "$pid" --server-log "$log" \
    --out "reports/tier2_${name}.json"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5
  return $status
}

# Qwen's card serves this checkpoint with `--quantization moe_wna16`; vLLM normally detects
# GPTQ on its own. If startup fails on quantization, rerun with QWEN_ARGS="--quantization
# moe_wna16". If it fails on a Mamba or CUDA-graph cache error, the vLLM recipe's fix is
# QWEN_ARGS="--max-cudagraph-capture-size 64".
# shellcheck disable=SC2086
run_model "$QWEN" ${QWEN_ARGS:-}
status_q=$?
# shellcheck disable=SC2086
run_model "$GEMMA" ${GEMMA_ARGS:-}
status_g=$?
echo "Tier 2 checks: Qwen exit $status_q, Gemma exit $status_g (0 = pass)."
# Keeping one passing model is enough to go on (design doc, Still uncertain item 4).
[ "$status_q" -eq 0 ] || [ "$status_g" -eq 0 ]
