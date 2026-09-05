# Contributing to TwinMind

This is a final-year capstone, so the git history is part of the deliverable: it
should read as a record of how the system was built. These conventions exist to
keep it that way.

## Branching

`main` stays green. Every commit on `main` passes `ruff check`, `mypy src/` and
`pytest`, and CI enforces that on every push.

Nothing is committed directly to `main`. Feature work happens on a branch named
`feat/<area>`, one branch per roadmap task:

| Branch | Scope |
| --- | --- |
| `feat/data-layer` | acquisition, loaders, preprocessing, EDA |
| `feat/model-health` | Model A — CNN-BiLSTM RUL and failure head |
| `feat/twin-sim` | the SimPy digital twin |
| `feat/surrogate` | Model C — heteroscedastic residual MLP |
| `feat/agents` | the LangGraph agent layer |
| `feat/dashboard` | the Streamlit dashboard |

Other prefixes follow the same shape when the work is not a roadmap task:
`fix/<area>`, `docs/<area>`, `chore/<area>`, `refactor/<area>`.

```bash
git switch -c feat/model-health main
```

Keep a branch to one roadmap task. If a second concern appears mid-branch — a
bug found in passing, a doc fix — land it on its own branch rather than smuggling
it into an unrelated one, so the history stays readable.

## Commits

Conventional commits throughout:

```
<type>(<scope>): <summary in the imperative, lower case, no full stop>

<body: what changed and, more importantly, why>
```

Types in use: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`,
`build`, `ci`, `chore`. Scope is the area — `data`, `utils`, `agents`,
`contracts`, `config`, `twin`, `surrogate`, `deps`.

```
feat(data): C-MAPSS preprocessing -- RUL, sensors, regimes, windows
fix(twin): stop the PM process double-counting downtime hours
test(agents): cover the rate limiter's token-bucket refill
```

The summary says what changed; the body says why. A reader six months from now —
or an examiner in a viva — needs the reasoning, not a restatement of the diff.
Where a decision was non-obvious (a threshold, a cap, a pin), record what forced
it.

## Merging

Squash-merge to `main`, with a conventional-commit title:

```bash
gh pr create --fill --base main
gh pr merge --squash --delete-branch
```

Squashing means one branch becomes one commit on `main`, so `git log main`
reads as a sequence of completed pieces of work rather than as a stream of
work-in-progress. The detail stays on the PR if it is ever needed.

Rewrite the squash-merge title to a proper conventional-commit line before
merging — GitHub defaults to the PR title plus a `(#N)` suffix, which is fine,
but the type and scope must be right.

## Releases

Tag `main` at the end of each roadmap week:

| Tag | Milestone |
| --- | --- |
| `v0.1-scaffold` | package structure, configs, contracts, LLM client |
| `v0.2-data` | acquisition, loaders, preprocessing, EDA |
| `v0.3-health` | Model A trained and evaluated |
| `v0.4-demand` | Model B trained and evaluated |
| `v0.5-twin` | SimPy digital twin driven by A and B |
| `v0.6-surrogate` | Model C emulating the twin |
| `v0.7-agents` | LangGraph agent layer end to end |
| `v0.8-dashboard` | Streamlit dashboard |
| `v1.0` | benchmark, ablations and write-up complete |

```bash
git tag -a v0.2-data -m "Data layer: acquisition, loaders, preprocessing, EDA"
git push origin v0.2-data
```

Annotated tags, not lightweight ones — they carry a date, an author and a
message, which is what makes them usable as evidence of when a milestone landed.

## Before you push

```bash
python -m pytest
```

```bash
python -m ruff check . && python -m mypy src/
```

Or let the hooks do it — see below.

## Pre-commit hooks

Installed once per clone:

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m pre_commit install
```

The hooks run on staged files at commit time:

- **ruff** — lint with `--fix`, then format.
- **trailing-whitespace**, **end-of-file-fixer** — keep diffs free of whitespace
  noise.
- **check-added-large-files** — 1 MB ceiling, so a stray parquet, checkpoint or
  figure cannot be committed by accident. Data and models are gitignored, but the
  ignore rules only help for paths that were anticipated; this catches the rest.
- **detect-private-key** — refuses a committed private key outright.
- **check-yaml**, **check-toml** — the configs are parsed by Pydantic at load
  time, so a malformed one should fail here rather than at 3am mid-training.
- **check-merge-conflict**, **mixed-line-ending** — Windows and CI must agree.

To run them over the whole tree rather than just staged files:

```bash
python -m pre_commit run --all-files
```

If a hook modifies a file, the commit aborts. Re-stage and commit again — that is
the hook working, not a failure.

Bypass with `--no-verify` only when you genuinely mean it, and say why in the
commit body.

## What never gets committed

`data/`, `models/`, `.env`, `kaggle.json`, checkpoints, `mlruns/` and the
generated PNGs under `reports/figures/` are all gitignored, and
`check-added-large-files` is the backstop.

Two specific things to be careful about:

- **`.env.example` holds placeholders only.** It is tracked, so a real value
  typed into it is a real leak. Put actual credentials in `.env`, which is
  ignored.
- **Favorita data cannot be redistributed.** It comes from a Kaggle competition
  governed by rules each user accepts individually. That, plus this being graded
  work, is why the repository is private.

Notebooks are committed with outputs cleared; `make eda` regenerates them.
