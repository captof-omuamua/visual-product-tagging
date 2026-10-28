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
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 78% | 61% (n=476) | 53% |
| Qwen3.5-4B, not fine-tuned | 86% | 60% (n=476) | 72% |
| Gemma 4 26B (Tier 2) | 76% | 65% (n=476) | 55% |

## pattern (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 87% | 81% (n=154) | 72% |
| Qwen3.5-4B, not fine-tuned | 86% | 80% (n=154) | 75% |
| Gemma 4 26B (Tier 2) | 84% | 79% (n=154) | 54% |

## style (n = 1094)

| Model | Agree | Seller | Cover@95 |
|---|---|---|---|
| Tier 1 v0 (Qwen3.5-4B, fine-tuned) | 77% | 48% (n=618) | 57% |
| Qwen3.5-4B, not fine-tuned | 69% | 47% (n=618) | 46% |
| Gemma 4 26B (Tier 2) | 76% | 47% (n=618) | 49% |

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
