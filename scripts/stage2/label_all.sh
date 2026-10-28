#!/usr/bin/env bash
# Label everything Tier 1 v0 needs, one model at a time on the GPU (~1.5-2 h, ₹0):
#   Gemma 4 26B (Tier 2):  train + dev + test  -> data/labels/gemma4.jsonl
#     train labels teach Tier 1; test answers are Tier 2's own score.
#   Qwen3-VL-8B (outside the cascade): dev + calibration + test -> data/labels/qwen3vl8b.jsonl
#     the independent, provisional answer key.
# Watch progress with:  tail -f reports/label_all.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here (no nvcc); see docs/stage1.md
PORT="${PORT:-8000}"
mkdir -p reports data/labels

run() {  # $1 model, $2 splits, $3 output name
  local model="$1" splits="$2" name="$3"
  local log="reports/vllm_label_${name}.log"
  echo "== ${model} on ${splits} (server log: ${log}) =="
  .venv-serve/bin/vllm serve "$model" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 6144 --gpu-memory-utilization 0.88 --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image": 3}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python scripts/stage2/label.py --model "$model" --splits "$splits" \
    --image-size 768 --server-pid "$pid" --out "data/labels/${name}.jsonl"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5
  return $status
}

run cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit train,dev,test gemma4; s1=$?
run Qwen/Qwen3-VL-8B-Instruct dev,calibration,test qwen3vl8b; s2=$?
echo "Labelling done: Gemma exit $s1, Qwen3-VL-8B exit $s2 (0 = ok)."
