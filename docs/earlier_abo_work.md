# Earlier work: ABO cascade kit (stage 1 and Tier 1 v0)

This was the project's first direction, on Amazon Berkeley Objects. It was set aside on 28 Oct 2026 because ABO's attribute values are typed by sellers and unchecked, so results could not be scored against trusted labels. The serving, labelling and fine-tuning code was reused for the ImplicitAVE work in the main README. Tier 1 v0 results: `reports/tier1_v0_eval.md`.


**Catalogue attributes at small-model cost: a calibrated small-to-large VLM cascade.**

Online catalogues need structured attributes, such as colour, material and pattern, for
millions of listings. Large vision-language models label them well but cost too much at
that scale. Flipkart's engineering team calls large-VLM API cost "a major hurdle" and
moved to compact 2–3B VLMs
([Flipkart Tech](https://blog.flipkart.tech/the-future-of-e-commerce-how-ai-is-learning-to-describe-products-with-less-data-8dfbf05f83a1)).

This project measures three things: how much of a large model's quality a fine-tuned
small VLM keeps, at what cost per 1,000 items, and when an item should escalate to the
large model. The escalation threshold comes from risk–coverage curves.

## Status

| Stage | Output | Status |
|---|---|---|
| 1. Tools, smoke test and data check | Qwen3.5-4B trained, merged and served on an RTX 5090; schema and log-prob checks; both large models timed; ABO types checked | Ready to run |
| 2. Task and data | Amazon Berkeley Objects subset, attribute schema, leakage-safe splits, gold set | Next |
| 3. Baselines | Title regex; zero-shot small and large VLMs | |
| 4. LoRA fine-tuning | Image plus title against title only; frozen against tuned vision layers; ranks 8–32 | |
| 5. Evaluation harness | Per-attribute F1, schema validity, hallucinated values, ECE, AURC | |
| 6. Serving and cascade | vLLM with the adapter, constrained JSON, calibrated escalation | |
| 7. Frontier and load | Cost per 1,000 items, p50 and p95 latency, throughput at concurrency 1–64 | |
| 8. Tracing | MLflow traces of escalations and schema failures | |
| 9. Write-up | Results, decision records, demo | |

## Run stage 1 (Ubuntu, NVIDIA GPU)

```bash
# Once: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: wget -qO- https://astral.sh/uv/install.sh | sh

git clone https://github.com/captof-omuamua/catalog-vlm-cascade.git
cd catalog-vlm-cascade
bash scripts/stage1/run_all.sh   # 1-2 hours the first time; writes reports/stage1_report.md
```

It checks the machine first and stops with the exact fix if something is missing. What
each check proves, the fixes, and the fallbacks are in [docs/stage1.md](docs/stage1.md).

After a PASS, commit the evidence:

```bash
git add reports/*.md reports/*.json requirements/
git commit -s -m "Stage 1: stack and data checks pass on RTX 5090"
git push
```

## Layout

```
src/catalog_cascade/   library code: load-test client, smoke checks, synthetic data
scripts/stage1/        the stage 1 runbook as scripts
tests/                 CPU tests, run in CI
docs/                  stage notes and decision records (docs/adr)
reports/               committed evidence from GPU runs
requirements/          exact versions frozen from a passing run
```

## Develop on CPU

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/ruff check .
```

Code is MIT-licensed. Data and model licences: [DATA_LICENSES.md](DATA_LICENSES.md).
