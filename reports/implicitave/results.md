# ImplicitAVE results (1,610 hand-checked test questions)

Score = % answered with the hand-checked value (= the paper's micro-F1). Cost = measured GPU time x an assumed rental price of $0.6/h for one GPU (₹95.46/$). Paper rows from Zou et al., ACL Findings 2024, Table 4 (no cost reported).

| Setup | Clothing | Footwear | Jewelry&GA | Food | Home | **All** | To big model | ₹ / 1,000 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B, zero-shot (ours) | 83.2 | 74.5 | 90.5 | 86.2 | 79.9 | **82.2** | 0% | 0.72 |
| Qwen3-VL-8B, zero-shot (ours) | 80.1 | 72.6 | 90.0 | 84.1 | 79.9 | **80.9** | 0% | 1.14 |
| Gemma 4 26B-A4B, zero-shot (ours) | 82.7 | 69.4 | 93.6 | 86.4 | 81.8 | **82.2** | 100% | 2.75 |
| Qwen3.5-4B, fine-tuned (ours) | 88.0 | 78.9 | 92.3 | 89.0 | 82.1 | **85.3** | 0% | 0.67 |
| Cascade: fine-tuned 4B -> Gemma when confidence < 0.57, threshold set on validation (ours) | 85.4 | 78.2 | 93.2 | 89.2 | 82.9 | **85.3** | 12% | 0.99 |
| GPT-4V (paper) | 77.4 | 81.4 | 90.5 | 90.8 | 89.9 | **86.8** | - | - |
| LLaVA-1.5-13B (paper) | 49.1 | 63.7 | 81.8 | 76.2 | 80.3 | **71.9** | - | - |
| Qwen-VL-7B (paper) | 59.7 | 57.7 | 84.1 | 76.9 | 74.0 | **70.9** | - | - |
| LAVIN, fine-tuned (paper's previous best) | 65.9 | 75.4 | 78.6 | 60.8 | 64.3 | **67.8** | - | - |

## Cascade: accuracy vs share sent to the big model

| Confidence threshold | Score (All) | Sent to Gemma | ₹ / 1,000 |
|---|---|---|---|
| 0.5 | 85.3 | 8% | 0.88 |
| 0.8 | 84.3 | 26% | 1.39 |
| 0.9 | 83.8 | 35% | 1.63 |
| 0.95 | 83.7 | 44% | 1.87 |
| 0.99 | 82.8 | 68% | 2.55 |
| 0.999 | 82.2 | 93% | 3.23 |

## Auto-accept or flag for review (fine-tuned 4B alone)

Keep the answers whose confidence clears a cut-off chosen on validation questions to hit the target accuracy; flag the rest for a person.

| Target accuracy | Cut-off | Test: auto-accepted | Test: accuracy of accepted |
|---|---|---|---|
| 90% | 0.53 | 90% | 90.2% |
| 95% | 0.72 | 79% | 94.6% |
| 98% | 0.92 | 62% | 97.5% |
