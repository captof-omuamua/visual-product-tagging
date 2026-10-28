#!/usr/bin/env bash
# Stage 1 local mini-pilot: Gemma 4 26B-A4B (768 px), then base Qwen3.5-4B (512 px),
# each on the 20 real pilot products. One model at a time on the GPU. Costs nothing.
# Watch progress with:  tail -f reports/local_pilot.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here; see docs/stage1.md
PORT="${PORT:-8000}"
mkdir -p reports

run() {  # $1 model, $2 image size, $3 short name, then extra vllm args
  local model="$1" size="$2" name="$3"; shift 3
  local log="reports/vllm_pilot_${name}.log"
  echo "== ${model} at ${size} px (server log: ${log}) =="
  .venv-serve/bin/vllm serve "$model" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 4096 --gpu-memory-utilization 0.88 --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image": 3}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' "$@" > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python scripts/stage1/local_pilot.py --model "$model" \
    --image-size "$size" --server-pid "$pid" --out "reports/local_pilot_${name}.json"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5  # let the GPU memory free before the next model
  return $status
}

run cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit 768 gemma4; s1=$?
run Qwen/Qwen3.5-4B 512 qwen35_4b; s2=$?
echo "Local pilot done: Gemma exit $s1, Qwen3.5-4B exit $s2 (0 = ok)."
