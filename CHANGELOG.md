# Changelog

A timestamped record of all notable changes. **Newest first.** Times are local
commit times (`git log --date=format:'%Y-%m-%d %H:%M'`).

> **Required:** every change to this repo must add an entry here in the same commit.
> Enforced by `.githooks/pre-commit`. See [CLAUDE.md](CLAUDE.md).

---

## 2026-06-26

### 08:52 — Manual opener minutes/role override layer
- `data_processing.load_overrides()` reads an optional `data/overrides.csv`
  (player_id or name + `minutes_mult`) and multiplies the final prediction — kept
  OUT of the model. The recommended way to fold in final friendly lineups, team
  news, World Cup fatigue at the opener. Applied in `main.py` and `serve.py`
  (exposed as an `override` flag per player; cache v5; ⚙ marker in the UI table).
- Added `overrides.example.csv` template, a regression test, and README docs. Tests 8/8.

### 08:48 — Surfaced the multi-GW planner in the web UI
- `serve.py`: precompute now builds 5-GW horizon projections + chip hints (cached);
  each player carries `horizon` + per-GW points; new `GET /api/plan?k=&free=&bank=`
  returns live transfer suggestions from the optimal squad. Cache version bumped to v4.
- `web/`: new "Multi-GW Planner" section — projected-points table (per GW + Σ),
  suggested transfers (with the −4 hit), and Triple Captain / Bench Boost GW hints.
- Verified via API + a Chrome-headless screenshot.

### 05:55 — Added changelog discipline
- Created this `CHANGELOG.md`, `CLAUDE.md` (working agreement), and a `.githooks/pre-commit`
  hook that blocks commits which don't update the changelog (`git config core.hooksPath .githooks`).
- Going forward, every change to this repo records a timestamped entry here.

## 2026-06-25

### 23:38 — Pipeline correctness-audit fixes + data-pipeline improvements (`ad50c5f`)
- **Fixed train/serve feature skew:** cold-start seeding was applied at forecast time
  but not training time. `build_feature_table` now seeds each row by games-played-so-far;
  `main.py`/`serve.py` seed both train and forecast with the same profiles. Added regression test.
- Fixed `next_unfinished_round` treating NaN `finished` as finished (`fillna(False)`).
- Opponent strength is now **venue-aware** (home/away split, not a blind average) → MAE 1.009 → 1.005.
- Hurdle fallback: a position with no points-regressor no longer predicts 0 for all.
- `model.predict` reindexes missing columns (no silent KeyError).
- Planner: metadata kept across blank GWs (OPT-1); transfer suggestions no longer crash on a
  current player with no fixture (OPT-2).
- serve cache fingerprint includes newest gw.csv mtime; `ingest_data` defaults to fresh fetch.
- **Added `history_past.csv` ingestion** (prior-season totals — bootstrap-static resets to 0 each season).
- **Added set-piece/penalty-taker features** (kept for the season opener; within-noise mid-season).
- Tests: 7/7.

### 22:54 — Multi-gameweek planner (`2d6ce6d`)
- `planner.py`: horizon projections (next N GWs, fixture-adjusted), transfer optimiser
  (−4 hit aware), chip hints. `optimizer.optimize_squad` gained transfer-planning params.

### 22:46 — Understat xG join, off by default (`6c9cf70`)
- `attach_understat()` + `compare_understat.py`. Measured within noise (FPL already carries
  Opta xG), so `use_understat` defaults False. Kept as a reproducible experiment.

### 22:39 — Web UX: data-source banner + instant-startup cache (`95533fa`)
- Banner clarifies when demo (not live) data is shown; `serve.py` caches precompute
  (`.cache/`) for ~instant restarts.

## 2026-06-21

### 22:55 — Web UI: live squad optimiser dashboard (`d98e0cc`)
- `serve.py` (stdlib server) + `web/` dashboard: optimal XI pitch, live re-optimise,
  player explorer, value scatter. `optimize_squad` gained lock/exclude.

## 2026-06-18

### 12:31 — Season-aware pipeline (`4a1669f`)
- Forecast the next *unfinished* gameweek; new-season opener mode (train on last season,
  cold-start GW1); graceful empty-data handling.

### 11:22 — Minutes/availability modelling (`2ffb8cd`)
- Live availability downweight (status / chance_of_playing). 3-band minutes hurdle tested
  (`compare_models.py`) but within noise, so the 2-part model stays default.

### 11:13 — Hyperparameter tuning (`3237cdb`)
- `tune.py` nested walk-forward search; adopted tuned defaults (~2.4% MAE gain on the locked test).

### 09:22 — Code-review fixes (`6c9b9c2`)
- LightGBM bagging/seed, DGW opponent averaging, qcut guard.

### 08:59 — Ground-up rebuild (`00425df`)
- Replaced the leaky LSTM prototype: official FPL API ingestion, leakage-free `(player, round)`
  features, position-specific hurdle gradient-boosted model, walk-forward validation vs baselines,
  `scipy.milp` squad optimiser, test suite.

---

## Pre-rebuild (context)
- The original repo trained a single LSTM on `[minutes, goals, assists]`. A review found its
  reported metrics were invalid (random splits on time-series, scaler fit before split,
  misaligned error report). That motivated the rebuild above.
