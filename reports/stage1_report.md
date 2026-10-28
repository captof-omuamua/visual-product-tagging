# Stage 1 report: PASS

Run at 2026-10-27 19:12 UTC on NVIDIA GeForce RTX 5090 Laptop GPU (23.9 GB, driver 595.99.02).

| Still uncertain (design doc) | What stage 1 found |
|---|---|
| 1. Qwen3.5-4B trains, merges and serves | trained 30 steps, loss 2.89 -> 0.0007; merged 8.68 GB; burst 200/200 ok; p50 10.694 s, p95 12.522 s; healthy after load: True; merged vs run-time answers identical: 1.0 |
| 2. Grammar enforced; log-probs as expected | forced schema obeyed 1.0; enum answers valid 1.0; structural tokens median log-prob 0.0, value tokens -0.0 |
| 3. Memory with 3 images of 512 px | training peak 9.38 GB (limit 22.0); serving burst 200/200 ok; p50 12.384 s, p95 15.141 s |
| 4. Local large models fit and are fast enough | Qwen/Qwen3.5-27B-GPTQ-Int4: FAILED, None req/s (~None h for 8,600 items), schema None; cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit: ok, 2.033 req/s (~1.2 h for 8,600 items), schema 1.0 |
| 5. ABO fill rates, families, duplicates, titles | SHOES 5749 (fails), CHAIR 1934 (fails), SOFA 1121 (fails), TABLE 905 (fails), RUG 763, LAMP 544 (fails), LIGHT_FIXTURE 515 |
| 6. Each field can reach its accepted-answer count | 1447 accepted per field (assumed); needs 2%: 322 (ok), 3%: 739 (ok), 4%: 2853 (no) |
| 7. Gemini tokens, thinking and image log-probs | not run |

## Warnings

- Qwen/Qwen3.5-27B-GPTQ-Int4 failed its check; the other one carries on.
- cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit used the slow Triton attention path.
- ABO types failing a check: SHOES, CHAIR, SOFA, TABLE, LAMP. Choose swaps from options B or C before stage 2.
- The Gemini mini-pilot did not run (set GEMINI_API_KEY).

## Next step

Commit reports/ and requirements/*-freeze.txt (`git add reports requirements && git commit -s`), then send this file to Claude to plan stage 2.
