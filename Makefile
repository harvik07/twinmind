# Every target is a one-line wrapper over a plain `python ...` command, so a
# Windows user without make can copy the command straight out of the README.

PYTHON ?= python

.PHONY: setup test test-fast lint format clean data data-optional eda notebooks \
        train-health train-demand rollouts train-surrogate benchmark app

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

eda:
	$(PYTHON) scripts/build_notebooks.py --execute

notebooks:
	$(PYTHON) scripts/build_notebooks.py

train-health:
	$(PYTHON) scripts/train_health.py --subset FD001

train-demand:
	$(PYTHON) scripts/train_demand.py --model nhits

rollouts:
	$(PYTHON) scripts/generate_rollouts.py --n 65000

train-surrogate:
	$(PYTHON) scripts/train_surrogate.py

benchmark:
	$(PYTHON) scripts/run_benchmark.py --method all

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
