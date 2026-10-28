#!/usr/bin/env bash
# Tier 1 v0 end to end, after scripts/stage2/label_all.sh (~2.5-3 h, ₹0):
#   1. fine-tune Qwen3.5-4B on Gemma's train labels and merge it
#   2. the merged model, then the untouched base model, answer the test split (512 px)
#   3. score everything against the Qwen3-VL-8B answer key
# Watch progress with:  tail -f reports/run_tier1.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit here (no nvcc); see docs/stage1.md
PORT="${PORT:-8000}"

answer_test() {  # $1 model path or id, $2 output name
  local model="$1" name="$2"
  local log="reports/vllm_test_${name}.log"
  echo "== ${model} answers the test split (server log: ${log}) =="
  .venv-serve/bin/vllm serve "$model" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 6144 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --limit-mm-per-prompt '{"image": 3}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "$log" 2>&1 &
  local pid=$!
  .venv-serve/bin/python scripts/stage2/label.py --model "$model" --splits test \
    --image-size 512 --server-pid "$pid" --out "data/labels/${name}.jsonl"
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 5
  return $status
}

echo "== Training Tier 1 v0 =="
.venv-train/bin/python scripts/tier1/train_tier1.py || { echo "Training failed"; exit 1; }
answer_test outputs/tier1-v0-merged tier1_v0
answer_test Qwen/Qwen3.5-4B qwen35_4b_base
.venv-serve/bin/python scripts/tier1/evaluate.py
