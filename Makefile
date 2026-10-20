# Every target is a one-line wrapper over a plain `python ...` command, so a
# Windows user without make can copy the command straight out of the README.

PYTHON ?= python

.PHONY: setup test test-fast lint format clean data data-optional eda notebooks \
        baselines train-health train-ai4i train-demand rollouts \
        train-surrogate benchmark app figures results-table

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not data"

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m mypy src/

format:
	$(PYTHON) -m ruff format . && $(PYTHON) -m ruff check --fix .

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','build','dist']]"

data:
	$(PYTHON) scripts/download_data.py --dataset all

data-optional:
	$(PYTHON) scripts/download_data.py --dataset all --include-optional

check-auth:
	$(PYTHON) scripts/download_data.py --check-auth

eda:
	$(PYTHON) scripts/build_notebooks.py --execute

notebooks:
	$(PYTHON) scripts/build_notebooks.py

baselines:
	$(PYTHON) scripts/train_baselines.py --subset all --seeds 3

train-health:
	$(PYTHON) scripts/train_health.py --subset all --seeds 3

train-ai4i:
	$(PYTHON) scripts/train_ai4i.py --model both --seeds 3

figures:
	$(PYTHON) scripts/make_figures.py

results-table:
	$(PYTHON) -m twinmind.eval.registry

train-demand:
	$(PYTHON) scripts/train_demand.py --model nhits

rollouts:
	$(PYTHON) scripts/generate_rollouts.py --n 65000

train-surrogate:
	$(PYTHON) scripts/train_surrogate.py

benchmark:
	$(PYTHON) scripts/run_benchmark.py --method all

app:
	$(PYTHON) -m streamlit run app/main.py
