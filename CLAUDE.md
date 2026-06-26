# CLAUDE.md — project working agreement

Instructions for any contributor (human or AI) working in this repo.

## ⚠️ Changelog discipline — REQUIRED

**Every change must add a timestamped entry to [CHANGELOG.md](CHANGELOG.md)** (newest
first: a `## YYYY-MM-DD` date header and a `### HH:MM — summary` entry), in the **same
commit** as the change, describing *what* changed and *why*.

- A `pre-commit` hook (`.githooks/pre-commit`) blocks commits that don't stage `CHANGELOG.md`.
- First-time setup in a fresh clone: `git config core.hooksPath .githooks`.
- Genuine exceptions (e.g. a non-code commit) may bypass with `git commit --no-verify`.

## What this project is

FPL points prediction + decision tool for the upcoming Premier League season.
Pipeline: `data_ingestion.py` (official FPL API) → `data_processing.py` (leakage-free
features) → `model.py` (position-specific hurdle gradient-boosted trees) →
`validation.py` (walk-forward) → `optimizer.py` / `planner.py` (ILP squad + transfers) →
`serve.py` + `web/` (dashboard). See [README.md](README.md).

## Engineering conventions (follow these)

1. **No leakage.** Features must be a function of past gameweeks only (or pre-match info).
   Validate with **walk-forward** keyed on the real `round`, never a random split.
2. **Measure before adopting.** New features/model changes must beat the baseline on
   walk-forward (see `compare_*.py`, `tune.py`); if within noise, keep the simpler option
   and record the experiment rather than shipping complexity.
3. **Train == serve.** A feature must be the identical function of history at training and
   forecast time (`build_feature_table` vs `build_upcoming_features`). Cold-start seeding is
   applied to both paths or neither.
4. **Keep tests green:** `python tests/test_pipeline.py` (currently 7/7).
5. Local Python here is miniconda (`/Users/bipan/miniconda3/bin/python3`) — has pandas/sklearn/scipy.
6. Don't wire World Cup / pre-season friendlies in as model features (noisy, no free xG since
   FBref dropped Opta in Jan 2026) — use only as a manual minutes/role overlay.
