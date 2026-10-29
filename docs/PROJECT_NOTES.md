# Project notes: status, decisions, how to resume

Last updated 29 Oct 2026. The README holds the results; this file holds the working context.

## Status
- **Done:** fine-tuned Qwen3.5-4B on ImplicitAVE: 85.3% on the 1,610 hand-checked test
  questions (GPT-4V 86.8, paper's best fine-tuned 67.8), ₹0.67 per 1,000. Auto-accept at a
  95% target: 79% accepted, 94.6% accurate. Published at
  github.com/captof-omuamua/visual-product-tagging.
- Outputs on the local machine (not in git): `outputs/iave-qwen35-4b-merged` (merged model),
  `outputs/iave-qwen35-4b-lora` (adapter), data under `data/raw/implicitave/`.

## Key decisions
| Decision | Why |
|---|---|
| Benchmark = ImplicitAVE, not ABO | ABO attribute values are seller-typed and unchecked; ImplicitAVE's test set is hand-checked and has published baselines |
| Small model = Qwen3.5-4B, LoRA r16, vision frozen, 1 epoch, ≤150 examples per value | Best zero-shot of the models tried (82.2); 48 min to train, 9.7 GB peak |
| Answers constrained to the allowed values (vLLM JSON schema), thinking off | No made-up values; confidence = probability of the chosen value's tokens |
| All cut-offs chosen on validation (2,000 rows), never on test | Avoids tuning on test; a test-tuned cascade would have shown 85.9 instead of 85.3 |
| Cascade to Gemma 4 26B reported but not used | After fine-tuning the 4B model beats Gemma (85.3 vs 82.2), so escalation adds cost, not accuracy |
| Cost = measured GPU time × assumed $0.60/h | Change `PRICE_USD_PER_HOUR` in `scripts/implicitave/analyze.py` |

## Known issues (machine/tooling)
- vLLM needs `VLLM_USE_FLASHINFER_SAMPLER=0` without a CUDA toolkit (all serving scripts set it).
- `Qwen/Qwen3.5-27B-GPTQ-Int4` runs out of memory at load on 24 GB (GPTQ config not applied);
  Gemma 4 26B-A4B AWQ works instead.
- Hugging Face Xet downloads stalled once; `HF_HUB_DISABLE_XET=1` fixed it.

## Ideas not done (if time allows)
- Close the gap to the "pick whichever model is right" ceiling (90.9% for the fine-tuned
  4B plus Gemma): a learned router, not raw confidence, might recover some of it.
- Multiple images per product, higher resolution, 2 epochs, a larger LoRA rank.
- A small FastAPI endpoint and a demo notebook.

## Earlier work
`docs/earlier_abo_work.md`, `reports/tier1_v0_eval.md`: the first direction on Amazon
Berkeley Objects (stage 1 tool checks, Tier 1 v0). Set aside on 28 Oct.
