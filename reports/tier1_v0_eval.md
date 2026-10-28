# Tier 1 v0: test split, provisional

Answer key: qwen3vl8b (outside the cascade). Agree = same value as the key; seller = same as the seller's mapped value; cover@95 = share answerable at 95% agreement, most confident first.

## color (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 92% | 87% (n=957) | 90% |
| Qwen3.5-4B, not fine-tuned | 89% | 77% (n=957) | 88% |
| Gemma 4 26B (Tier 2) | 91% | 84% (n=957) | 89% |

## material (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 92% | 76% (n=476) | 94% |
| Qwen3.5-4B, not fine-tuned | 94% | 78% (n=476) | 99% |
| Gemma 4 26B (Tier 2) | 90% | 77% (n=476) | 86% |

## pattern (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 93% | 87% (n=154) | 94% |
| Qwen3.5-4B, not fine-tuned | 90% | 86% (n=154) | 84% |
| Gemma 4 26B (Tier 2) | 90% | 88% (n=154) | 72% |

## style (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 87% | 87% (n=618) | 64% |
| Qwen3.5-4B, not fine-tuned | 86% | 84% (n=618) | 51% |
| Gemma 4 26B (Tier 2) | 85% | 85% (n=618) | 57% |

## shape (n = 184)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 95% | 82% (n=127) | 99% |
| Qwen3.5-4B, not fine-tuned | 95% | 84% (n=127) | 100% |
| Gemma 4 26B (Tier 2) | 97% | 86% (n=127) | 100% |

## finish (n = 122)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 78% | 82% (n=51) | 35% |
| Qwen3.5-4B, not fine-tuned | 88% | 86% (n=51) | 72% |
| Gemma 4 26B (Tier 2) | 77% | 82% (n=51) | 50% |
