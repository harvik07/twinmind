# TwinMind

An agentic deep-learning digital twin of a manufacturing line.

The system learns how the line *degrades* and how demand *moves*, simulates the
line under those learned dynamics, learns a fast emulator of that simulation, and
puts an LLM agent in front of the whole thing to search for better operating
policies — while keeping every reported number deterministic.

## Architecture

1. **Model A — machine health.** CNN-BiLSTM on NASA C-MAPSS predicting remaining
   useful life, plus a failure-probability head trained on UCI AI4I 2020.
   MC-dropout gives predictive uncertainty.
2. **Model B — demand.** N-HiTS / PatchTST via Nixtla `neuralforecast` on Kaggle
   Favorita store sales, trained with a multi-quantile loss at q10/q50/q90.
3. **Digital twin.** A SimPy discrete-event simulation of machines, buffers,
   shifts, inventory and suppliers. Its failure hazard comes from Model A and its
   order book from Model B — no hard-coded MTBF anywhere.
4. **Model C — surrogate.** A heteroscedastic residual MLP trained on ~65k
   simulator rollouts that emulates the twin roughly 10,000× faster, so an agent
   can screen hundreds of scenarios in seconds and re-verify only the top
   candidates on the real twin.
5. **Agent layer.** A LangGraph state machine — Planner / Explorer / Evaluator /
   Critic / Explainer — driven by Groq.
   **Hard rule:** the LLM only proposes scenario parameters and writes
   explanations. Every KPI, comparison and constraint check is deterministic
   Python.
6. **Dashboard and baselines.** A Streamlit dashboard, plus Optuna TPE / CMA-ES
   and random / grid search as evaluation baselines.

Everything runs CPU-only except the training scripts, which are written to be
importable from a Kaggle or Colab notebook that clones this repo.

## Setup — Windows (development)

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -e ".[dev]"
```

```bash
copy .env.example .env
```

Fill in `GROQ_API_KEY` (and `KAGGLE_USERNAME` / `KAGGLE_KEY` if you want
`download_data.py` to work). Then check the scaffold is sound:

```bash
python -m pytest
```

The demand model needs a heavy extra dependency tree, so it is optional:

```bash
python -m pip install -e ".[forecast]"
```

## Setup — Kaggle / Colab (training)

```python
!git clone https://github.com/<you>/TwinMind.git
%cd TwinMind
!pip install -q -r requirements.txt
!pip install -q -e .
```

`requirements.txt` and `requirements-dev.txt` are flat files generated from
`pyproject.toml`, because Kaggle notebooks install from a flat file. Add
`!pip install -q -r requirements-forecast.txt` only when training Model B.

Then, in a notebook cell:

```python
from twinmind.utils.config import load_config
from twinmind.utils.seed import set_seed
set_seed(42)
config = load_config("models")
```

## Commands

Every Makefile target is a one-line wrapper over a plain `python` command. On
Windows without `make`, run the raw command instead.

| Target | Raw command |
| --- | --- |
| `make setup` | `python -m pip install -e ".[dev]"` |
| `make test` | `python -m pytest` |
| `make lint` | `python -m ruff check . && python -m mypy src/` |
| `make format` | `python -m ruff format . && python -m ruff check --fix .` |
| `make clean` | removes `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `build`, `dist` |
| `make data` | `python scripts/download_data.py --dataset all` |
| `make train-health` | `python scripts/train_health.py --subset FD001` |
| `make train-demand` | `python scripts/train_demand.py --model nhits` |
| `make rollouts` | `python scripts/generate_rollouts.py --n 65000` |
| `make train-surrogate` | `python scripts/train_surrogate.py` |
| `make benchmark` | `python scripts/run_benchmark.py --method all` |
| `make app` | `python -m streamlit run app/streamlit_app.py` |

On Windows, `make lint` splits into two commands:

```bash
python -m ruff check .
```

```bash
python -m mypy src/
```

## Layout

```
configs/     data.yaml, twin.yaml, models.yaml, agent.yaml — validated by Pydantic
data/        raw/ interim/ processed/ (gitignored)
src/twinmind/
  data/      dataset loaders, windowing, feature engineering
  models/    Model A (health), Model B (demand), Model C (surrogate)
  twin/      SimPy machines, inventory, demand process, costs, runner + contracts
  optimise/  random / grid / Optuna baselines and the oracle
  agents/    LangGraph graph, nodes, deterministic tools, memory, LLM client
  eval/      metrics, ablations, explanation faithfulness
  utils/     paths, config, seeding, logging
notebooks/   01_..07_ exploration and Kaggle training (see notebooks/README.md)
app/         Streamlit dashboard
scripts/     CLI entry points for data, training and benchmarking
tests/       pytest suite
reports/     metrics, LLM call logs, figures/
```

## Design notes

**Configs are typed.** Each YAML in `configs/` maps to one Pydantic model with
`extra="forbid"`. A mistyped key raises `ValidationError` at load time rather
than being silently ignored and surfacing as a wrong number mid-training.
`${VAR}` and `${VAR:-default}` are interpolated from the environment.

**Paths are resolved from `__file__`, not the cwd.** `PROJECT_ROOT` is the same
whether you run from the repo root, from `tests/`, or from a notebook, and every
accessor creates its directory on first use.

**The scenario space is 18-dimensional** and declared once, in `twin.yaml`:
throughput multiplier, shift count, overtime hours, a PM interval and a speed
setpoint for each of the five machines, supplier choice, reorder point, order
quantity, safety stock, and price multiplier. `SCENARIO_DIMS` in
`twin/runner.py` pins the canonical vector order that the surrogate, the
optimisers and the agent all share; a test asserts the two agree.

**Corrective maintenance costs exactly 5× preventive**, and a config validator
enforces that ratio — it is the asymmetry the whole maintenance-policy argument
rests on, so it should not be quietly editable.

**The LLM is rate-limited and logged.** `GroqClient` enforces 28 requests/min and
5,500 tokens/min (Groq's free tier is ~30 RPM / 6k TPM), retries twice on schema
violation, and appends model, token counts and latency to
`reports/llm_calls.jsonl` for the calls-per-query metric. `EchoClient` is a
deterministic offline stub so the test suite and CI never touch the network.

## Current status

**Scaffold complete — no models, no simulator, no agent yet.**

| Component | Status |
| --- | --- |
| Repo layout, packaging, CI | Done |
| `utils/` — paths, config, seed, logging | Done |
| `agents/llm_client.py` — Groq + Echo, rate limiter, call log | Done |
| Data contracts — `Scenario`, `KPIVector`, `KPIWithCI` | Done |
| `configs/` — all four YAMLs with real values | Done |
| Test suite (88 tests) | Passing |
| Model A — machine health | Stubs only |
| Model B — demand | Stubs only |
| Digital twin (SimPy) | Stubs only |
| Model C — surrogate | Stubs only |
| Agent layer (LangGraph nodes) | Stubs only |
| Streamlit dashboard | Placeholder shell |

Every stub raises `NotImplementedError` with the function's name, so nothing
fails silently as it gets filled in.
