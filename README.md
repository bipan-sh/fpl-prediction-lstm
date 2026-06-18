# fpl-prediction

Predicts Fantasy Premier League (FPL) points for the next gameweek and turns those
predictions into an optimal squad, starting XI, and captain under the real FPL rules.

This is a ground-up rebuild of an earlier LSTM prototype. The rebuild was driven by
two findings: (1) the old data source (a community GitHub mirror) stopped its weekly
updates after 2024-25, and (2) the old model's reported accuracy was an artifact of
data leakage (random splits on time-series, scaler fit on the whole dataset, an error
report aligned to the wrong rows). The state of the art on free data uses
**position-specific gradient-boosted trees with a hurdle structure**, not LSTMs.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Ingestion | `data_ingestion.py` | Pulls live data from the **official FPL API** (bootstrap-static, fixtures, element-summary) with on-disk caching. |
| 2. Features | `data_processing.py` | Builds a **leakage-free** `(player, round)` table: real `round` time axis, double-gameweek aggregation, lagged/rolling form, xG/xA, ICT, BPS, fixture & opponent strength. |
| 3. Model | `model.py` | **Position-specific hurdle model**: `P(player appears) × E[points | appeared]`, per position, using gradient-boosted trees. |
| 4. Validation | `validation.py` | **Walk-forward (rolling-origin)** evaluation vs naive baselines, with rank/captaincy metrics — the only honest way to score a forecaster. |
| 5. Tuning | `tune.py` | **Nested walk-forward** hyperparameter search: an inner walk-forward picks the model knobs; a locked, never-seen final-test block reports the honest gain. |
| 6. Optimization | `optimizer.py` | **Integer linear program** (`scipy.optimize.milp` / HiGHS) for the 15-man squad, XI, and captain under budget / quota / max-3-per-club / formation rules. |
| 7. Orchestration | `main.py` | Runs evaluation, trains the final model, predicts the next round, and prints the optimal squad. |

## Why these choices

- **Gradient-boosted trees, not LSTM** — match paid projection services on free data, are scale-invariant (no scaler-leakage), and handle missing early-season form natively.
- **Hurdle structure** — the target is ~50% zeros (benched players). Modelling "will they play" separately is the single biggest accuracy lever in FPL.
- **Walk-forward validation** — train on rounds `< t`, test on round `t`. Naive baselines (predict-last, trailing mean, season mean) are mandatory: a model that can't beat them has no edge.
- **Availability downweight** — for a live forecast, predictions are scaled by the FPL API's current `status` / `chance_of_playing_next_round`, so injured/suspended/doubtful players are correctly demoted. This is the part the historical model can't learn (it never sees injury news).

### Experiments kept as evidence (not in the default path)
- `tune.py` — nested walk-forward hyperparameter search (its tuned config *is* the current default; ~2.4% MAE gain on the locked test).
- `compare_models.py` + `MinutesHurdleModel` — a finer 3-band minutes hurdle (DNP / cameo / start). Tested head-to-head it was only ~0.8% better (within noise), so the simpler 2-part model remains the default. Kept so the decision is reproducible and the model is ready if more data shifts the verdict.

## Requirements

```
pip install pandas numpy scipy scikit-learn requests joblib
# optional, faster model engine (falls back to scikit-learn HistGradientBoosting):
pip install lightgbm
```

## How to run

```bash
python data_ingestion.py     # download current-season data (needs network)
python main.py               # build features, evaluate, predict, optimize squad
```

`main.py` works on whatever data is under `data/` (per-player `gw.csv` files with a
`round` column, plus `players_raw.csv` and `teams.csv`).

## Backtest vs. forecast

`main.py` does two distinct things, and it is important not to confuse them:

- **Backtest** (the walk-forward table): predicts *past, completed* gameweeks so it
  can compare predictions against actual points. This is how accuracy is measured.
- **Forecast** (the "Top 15 / squad" section): predicts the **next, not-yet-played**
  gameweek (`build_upcoming_features`) using each player's form so far plus that
  round's known fixtures. There are no actual points for this round — it is a real
  forecast, not a re-scored past gameweek.

The bundled `data/` is a frozen 2024-25 snapshot ending at round 26, so the demo
forecasts round 27. With live data, the model forecasts whatever the next gameweek is.

## Brand-new season (cold start) — implemented

At the very start of a season (GW1–~GW3) there is **no within-season form yet**, so the
form features would all be empty. `build_prior_profiles()` + `seed_cold_start()` handle
this by seeding early-gameweek features from the **previous season's per-game profile**,
blended by a shrinkage weight `w = n / (n + k0)` (n = games played this season): 100%
prior at GW1, fading to within-season form as games accumulate. Players with no prior FPL
history — new signings, promoted clubs, youth — fall back to the average profile of their
**position × price bucket**.

Enable it by pointing at last season's ingested data:

```bash
FPL_PRIOR_SEASON_DIR=data_2025_26 python main.py
```

(FPL only publishes a new season's players/prices/fixtures ~2 weeks before kickoff, so
there is nothing to forecast until then.)

## Roadmap

- Multi-gameweek transfer planning (free transfers, the −4 hit, price changes).
- Chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain) as decision variables.
- Add the FPL API's own `ep_next` as an extra baseline once ingested live.
- Join Understat xG/xA via a curated FPL↔Understat player-id map.
