"""
Does adding Understat advanced stats (npxG / shots / key_passes / xGChain) improve
the model? Measured the honest way: identical walk-forward, with vs without.

Run:  python compare_understat.py
"""
from __future__ import annotations

import logging

from data_processing import build_feature_table, feature_columns
from model import make_predictor
from validation import walk_forward_predict, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_TEST, MIN_TRAIN = 6, 8


def _score(use_understat):
    t = build_feature_table("data", use_understat=use_understat)
    feats = feature_columns(t)
    rounds = sorted(t["round"].unique())[-N_TEST:]
    m = compute_metrics(walk_forward_predict(t, make_predictor(feats), rounds, MIN_TRAIN))
    return m, len(feats)


def main():
    base, nb = _score(False)
    us, nu = _score(True)
    logger.info("\n===== Understat features: off vs on (walk-forward, last %d rounds) =====", N_TEST)
    logger.info("  %-16s MAE=%.4f RMSE=%.4f spearman=%.4f  (%d features)", "FPL only", base["MAE"], base["RMSE"], base["spearman"], nb)
    logger.info("  %-16s MAE=%.4f RMSE=%.4f spearman=%.4f  (%d features)", "FPL + Understat", us["MAE"], us["RMSE"], us["spearman"], nu)
    dmae = base["MAE"] - us["MAE"]
    logger.info("  delta: MAE %+.4f (%+.1f%%), spearman %+.4f", dmae, 100 * dmae / base["MAE"], us["spearman"] - base["spearman"])
    if dmae > 0.01 or us["spearman"] - base["spearman"] > 0.01:
        logger.info("  VERDICT: Understat helps — keep it on.")
    elif dmae < -0.01:
        logger.info("  VERDICT: Understat hurts — leave it off.")
    else:
        logger.info("  VERDICT: within noise.")


if __name__ == "__main__":
    main()
