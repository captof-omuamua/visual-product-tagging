#!/usr/bin/env bash
# Zero-shot (no fine-tuning) answers on the 1,610 ImplicitAVE test questions, one model at a
# time on the GPU (~20 min in all, ₹0). Results in reports/implicitave/<name>.json.
# Usage: bash scripts/implicitave/zero_shot.sh                  (all three models)
#        bash scripts/implicitave/zero_shot.sh MODEL NAME [ARGS] (one model; ARGS go to
#        predict.py, e.g. --split val --limit 1500)
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here (no nvcc); see docs/stage1.md
PORT="${PORT:-8000}"
mkdir -p reports/implicitave

run() {  # $1 model id or path, $2 short name, then extra predict.py args
  local model="$1" name="$2"; shift 2
  local log="reports/implicitave/vllm_${name}.log"
  echo "== ${model} (server log: ${log}) =="
  .venv-serve/bin/vllm serve "$model" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 4096 --gpu-memory-utilization 0.88 --max-num-seqs 32 \
    --limit-mm-per-prompt '{"image": 1}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python scripts/implicitave/predict.py --model "$model" \
    --server-pid "$pid" --workers 32 --out "reports/implicitave/${name}.json" "$@"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5
  return $status
}

if [ "$#" -ge 2 ]; then
  run "$@"
else
  run Qwen/Qwen3.5-4B qwen35_4b_zeroshot
  run Qwen/Qwen3-VL-8B-Instruct qwen3vl_8b_zeroshot
  run cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit gemma4_26b_zeroshot
fi
