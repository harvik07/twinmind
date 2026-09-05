# Notebooks

Notebooks are for exploration and for running training on Kaggle/Colab GPUs.
Anything that needs to be reproducible belongs in `src/twinmind/` or `scripts/`;
a notebook should import from the package, never redefine logic.

## Naming

`NN_short_topic.ipynb`, numbered in the order the project is meant to be read:

| Prefix | Topic |
| --- | --- |
| `01_` | Data exploration -- C-MAPSS, AI4I 2020, Favorita |
| `02_` | Model A: machine health (RUL + failure probability) |
| `03_` | Model B: demand forecasting |
| `04_` | Digital twin: SimPy line validation |
| `05_` | Rollout generation and Model C surrogate |
| `06_` | Agent runs and ablations |
| `07_` | Results, figures and the benchmark write-up |

Commit notebooks with outputs cleared.

## Running on Kaggle

```python
!git clone https://github.com/<you>/TwinMind.git
%cd TwinMind
!pip install -q -r requirements.txt
!pip install -q -e .
```

Then `from twinmind.utils.config import load_config` works as it does locally.
Add `-r requirements-forecast.txt` only for the demand model.
