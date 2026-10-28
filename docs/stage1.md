# Stage 1: tools, smoke test and data check

Stage 1 proves the stack and the data hold up on the RTX 5090 before any project code
depends on them. Budget: 6 hours, including fallbacks. It answers the first seven
"Still uncertain" items in the design doc; `reports/stage1_report.md` has one row each.

## Before you start (from Windows)

1. Reboot into Ubuntu.
2. Check the driver: `nvidia-smi`. The banner must say `CUDA Version: 13.0` or higher.
   If not: `sudo ubuntu-drivers install`, reboot, check again.
3. Free disk: about 110 GB on the drive holding this folder and `~/.cache/huggingface`.
4. Install uv once: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new
   terminal.
5. Optional, for step 7: a Gemini API key from aistudio.google.com. Keep it out of files:
   `export GEMINI_API_KEY=...` in the terminal only.

Then:

```bash
cd catalog-vlm-cascade
bash scripts/stage1/run_all.sh            # everything; 1-2 hours the first time
SKIP_TIER2=1 bash scripts/stage1/run_all.sh   # without the two large models (faster)
```

Each step writes its own report even if it fails, and later steps still run.

## What runs

| Step | Script | What it proves | Pass condition |
|---|---|---|---|
| 1. Environment | `check_env.py` | Linux, a driver new enough for vLLM's CUDA 13.0 wheels, uv, about 110 GB free | No blockers |
| 2. Environments | `setup_envs.sh` | vLLM 0.32.0, Unsloth (with open_clip) and google-genai install side by side (ADR 0001) | All install; freeze files written |
| 3. ABO data check | `abo_check.py` | English fill per field, brand families, duplicates and title lengths for the six types; a first estimate of whether each field can be certified; 20 pilot products | Every type passes its three checks |
| 4. Training | `train_smoke.py` | Qwen3.5-4B trains with 16-bit LoRA, vision frozen, 3 images of 512 px per item; peak memory; the thinking switch; merge to 16-bit | The loss falls, peak ≤ 22 GB, the merged model saves |
| 5. Serving | `serve_smoke.sh` | Phase A: base model plus run-time adapter, burst of 200 and sustained load. Phase B: merged model, burst, schema enforcement and log-probs, answers compared with phase A | Every request succeeds, the server stays healthy, the forced schema is obeyed 100% |
| 6. Large models | `tier2_smoke.sh` | Qwen3.5-27B (GPTQ Int4) and Gemma 4 26B-A4B (AWQ) each load at 4k context with 3 images of 768 px; speed; schema enforcement; slow-attention fallback | At least one passes |
| 7. Gemini mini-pilot | `gemini_pilot.py` | Real token and thinking counts, whether log-probs come back for images, JSON rate; capped at $0.60 | Runs; numbers recorded |
| Report | `report.py` | `reports/stage1_report.md` | PASS |

## Why these checks

- **Forced schema.** The schema allows only nonsense words ("xq-zebra"). If vLLM enforces
  it, every answer must use them. Gemma 4 once skipped enforcement silently
  ([vLLM #39130](https://github.com/vllm-project/vllm/issues/39130)), so this runs on every
  model, and Gemma is served without `--reasoning-parser`.
- **Log-probs under a grammar.** Structural tokens should score about 0, because vLLM masks
  disallowed tokens first. Only value tokens can carry confidence (design doc, hand-off rule).
- **Merged against run-time adapter.** Tier 1 is served merged. A run-time LoRA on Qwen's
  Gated DeltaNet layers hung in one 2026 report, so phase A exercises it once and the two
  servings' answers are compared at temperature 0.
- **Thinking.** Qwen3.5's template thinks unless `enable_thinking=False`. The server sets it
  off with `--default-chat-template-kwargs`, and the training report shows whether the
  training text matches that format. If it doesn't, stage 5 renders training text itself.
- **A burst of 200 with `--max-num-seqs 64`.** More than vLLM schedules at once, so the
  queueing path runs too. An RTX 5090 user reported a deadlock after about 50 requests on
  vLLM 0.11.2 ([vLLM forum](https://discuss.vllm.ai/t/bug-segfault-in-pythonsymnodeimpl-and-deadlock-on-rtx-5090-blackwell-with-vllm-0-11-2/2688)).

## Run one step on its own

```bash
python3 scripts/stage1/check_env.py
bash scripts/stage1/setup_envs.sh
.venv-train/bin/python scripts/stage1/abo_check.py
.venv-train/bin/python scripts/stage1/train_smoke.py
bash scripts/stage1/serve_smoke.sh
bash scripts/stage1/tier2_smoke.sh
GEMINI_API_KEY=... .venv-api/bin/python scripts/stage1/gemini_pilot.py   # after abo_check
python3 scripts/stage1/report.py
```

## If something fails

| Symptom | Likely cause | Fix |
|---|---|---|
| `BLOCKER: Your driver supports CUDA 12.x` | vLLM 0.32's wheels need CUDA 13.0 | `sudo ubuntu-drivers install` (or `nvidia-driver-580-open` or newer), reboot, rerun |
| Setup or `import unsloth` fails with a torch or CUDA mismatch | The torch build uv picked | `TRAIN_TORCH_BACKEND=cu128 bash scripts/stage1/setup_envs.sh` |
| Training fails at `load_model` with "no kernel image" | torch built without sm_120 | Check `gpu.arch_list` in `reports/train_smoke.json` includes `sm_120`; reinstall a CUDA 12.8+ build |
| Training fails at `train` with out-of-memory | 3 images of 512 px are too much | `.venv-train/bin/python scripts/stage1/train_smoke.py --batch-size 1`, then `--image-size 448` or `--images-per-item 2` |
| Training fails at `trainer`, `train` or `merge` | Unsloth or TRL API drift | Send the traceback in `reports/train_smoke.json`; usually a one-line fix |
| Qwen3.5-4B still fails after 2.5 hours | The toolchain for this model | Fall back to Qwen3-VL-8B: `--model Qwen/Qwen3-VL-8B-Instruct --load-in-4bit` and `MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/stage1/serve_smoke.sh` |
| vLLM warm-up fails: `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | FlashInfer compiles its sampling kernel at warm-up, and only the driver (no CUDA toolkit) is installed | The serving scripts now set `VLLM_USE_FLASHINFER_SAMPLER=0`; install the CUDA 13 toolkit only if another FlashInfer error appears |
| Qwen3.5-27B-GPTQ-Int4: CUDA out of memory in `create_weights` before its weights download | vLLM reserves 16-bit space for many layers (GPTQ config not applied); `--quantization moe_wna16` gives the same error | Not fixed. Gemma 4 26B-A4B AWQ is local Tier 2 instead (decision of 28 Oct) |
| vLLM exits during startup | See `reports/vllm_*.log` | Try `GPU_UTIL=0.75`; then the Docker image below |
| Requests time out, or the server is stuck after the load | The deadlock class of bug | Keep the log; try the Docker image; then SGLang |
| `grammar_enforced` fails | Structured output is off or bypassed | Send `reports/serve_merged.json` and the log; nothing in stage 5 should run until it passes |
| Qwen3.5-27B won't start | Quantisation or cache settings | `QWEN_ARGS="--quantization moe_wna16" bash scripts/stage1/tier2_smoke.sh` (as on its model card), or `QWEN_ARGS="--max-cudagraph-capture-size 64"` for Mamba-cache errors |
| A large model runs out of memory | 3 images of 768 px at 4k context | `GPU_UTIL=0.95`, or lower the image size for that model |
| `abo_check.py` can't download | Network or proxy | Download `archives/abo-listings.tar` from the ABO page by browser into `data/raw/abo/` and rerun |

Time cap: if the toolchain still fails after one 2.5-hour session, switch the serving path
or the model rather than debugging further.

Docker fallback (needs the NVIDIA Container Toolkit). The image's entrypoint is `vllm serve`:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD/outputs:/outputs" \
  vllm/vllm-openai:v0.32.0 /outputs/smoke-merged --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 3}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
# in a second terminal:
.venv-serve/bin/python -m catalog_cascade.smoke --model /outputs/smoke-merged --grammar
```

Use `vllm/vllm-openai:v0.32.0-cu129` if the driver supports CUDA 12.9 but not 13.0.

## What to say about it in an interview

- "Before writing project code, I validated the exact setup on the target GPU: the model,
  the merge, three images per request, and a 200-request burst."
- "The system's confidence signal depends on how vLLM computes log-probs under a grammar,
  so I tested that directly instead of assuming it."
- "I check schema enforcement with a schema the model would never satisfy by accident,
  because one model family has silently skipped it."
