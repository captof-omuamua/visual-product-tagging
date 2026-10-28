.PHONY: stage1 check setup abo-check train-smoke serve-smoke tier2-smoke pilot report test lint

stage1:  ## Run all of stage 1: check, set up, data, train, serve, large models, pilot, report
	bash scripts/stage1/run_all.sh

check:
	python3 scripts/stage1/check_env.py

setup:
	bash scripts/stage1/setup_envs.sh

abo-check:
	.venv-train/bin/python scripts/stage1/abo_check.py

train-smoke:
	.venv-train/bin/python scripts/stage1/train_smoke.py

serve-smoke:
	bash scripts/stage1/serve_smoke.sh

tier2-smoke:
	bash scripts/stage1/tier2_smoke.sh

pilot:  ## needs GEMINI_API_KEY in the environment
	.venv-api/bin/python scripts/stage1/gemini_pilot.py

report:
	python3 scripts/stage1/report.py

test:  ## CPU-only tests; run inside any environment with the dev extras
	python -m pytest -q

lint:
	ruff check .
