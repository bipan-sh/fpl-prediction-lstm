"""
Nested walk-forward hyperparameter tuning for the FPL points model.

Goal: find out whether tuning beats the current hard-coded hyperparameters, WITHOUT
fooling ourselves. The scheme keeps the data that PICKS the hyperparameters strictly
separate from the data that REPORTS the final score:

    rounds .......................................  (real gameweeks, ascending)
    [        tuning data        ][   FINAL TEST   ]
            |                            |
            | inner walk-forward         | locked: touched once, with the winning
            | scores each candidate      | config AND the current default, head-to-head

So the FINAL TEST is never seen while choosing knobs -> the comparison is honest.

Run:  python tune.py
"""
from __future__ import annotations

import logging
import random

import numpy as np

from data_processing import build_feature_table, feature_columns
from model import make_predictor, DEFAULT_PARAMS, engine_name
from validation import walk_forward_predict, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Search space (canonical, engine-agnostic names; see model.DEFAULT_PARAMS).
SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaves": [15, 31, 63],
    "n_trees": [200, 300, 400],
    "min_leaf": [20, 50, 100],
    "l2": [0.0, 1.0, 5.0],
}
N_CANDIDATES = 15           # random configs to try
N_FINAL_TEST_ROUNDS = 3     # most recent rounds, locked away for the honest comparison
N_INNER_ROUNDS = 3          # rounds used inside the tuning data to score candidates
MIN_TRAIN_ROUNDS = 8
SEED = 42


def _sample_configs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, configs = set(), []
    while len(configs) < n:
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def _score(df, params, test_rounds) -> dict:
    preds = walk_forward_predict(df, make_predictor(feature_columns(df), params),
                                 test_rounds, MIN_TRAIN_ROUNDS)
    return compute_metrics(preds)


def main() -> None:
    table = build_feature_table("data")
    rounds = sorted(table["round"].unique())
    final_test = rounds[-N_FINAL_TEST_ROUNDS:]
    tuning_max = final_test[0] - 1
    tuning_df = table[table["round"] <= tuning_max].copy()
    inner_rounds = sorted(tuning_df["round"].unique())[-N_INNER_ROUNDS:]

    logger.info("Engine: %s", engine_name())
    logger.info("FINAL TEST rounds (locked): %s", final_test)
    logger.info("Tuning data: rounds <= %d | inner scoring rounds: %s", tuning_max, inner_rounds)

    # --- Inner search: score each candidate on the tuning data only ---
    configs = _sample_configs(N_CANDIDATES, SEED)
    results = []
    for i, cfg in enumerate(configs, 1):
        mae = _score(tuning_df, cfg, inner_rounds)["MAE"]
        results.append((mae, cfg))
        logger.info("  candidate %2d/%d  inner MAE=%.4f  %s", i, len(configs), mae, cfg)
    results.sort(key=lambda r: r[0])
    best_mae, best_cfg = results[0]
    default_inner = _score(tuning_df, DEFAULT_PARAMS, inner_rounds)["MAE"]
    logger.info("Best inner MAE=%.4f with %s (default inner MAE=%.4f)",
                best_mae, best_cfg, default_inner)

    # --- Honest head-to-head on the locked FINAL TEST ---
    cur = _score(table, DEFAULT_PARAMS, final_test)
    tuned = _score(table, best_cfg, final_test)

    logger.info("\n===== FINAL TEST (rounds %s) — current vs tuned =====", final_test)
    logger.info("  %-8s MAE=%.4f RMSE=%.4f spearman=%.4f", "current", cur["MAE"], cur["RMSE"], cur["spearman"])
    logger.info("  %-8s MAE=%.4f RMSE=%.4f spearman=%.4f", "tuned", tuned["MAE"], tuned["RMSE"], tuned["spearman"])
    delta = cur["MAE"] - tuned["MAE"]
    pct = 100 * delta / cur["MAE"] if cur["MAE"] else 0.0
    logger.info("  tuned best config: %s", best_cfg)
    if delta > 0.01:
        logger.info("  VERDICT: tuning improves MAE by %.4f (%.1f%%). Worth adopting.", delta, pct)
    elif delta < -0.01:
        logger.info("  VERDICT: tuning is WORSE by %.4f. Keep current defaults.", -delta)
    else:
        logger.info("  VERDICT: within noise (delta %.4f). Current defaults are fine.", delta)


if __name__ == "__main__":
    main()
