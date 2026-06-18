"""
Head-to-head: current 2-part hurdle vs. the 3-band minutes hurdle.

Same discipline as tune.py — judge the structural change on identical walk-forward
splits, so we adopt it only if it genuinely helps (not on a hunch).

Run:  python compare_models.py
"""
from __future__ import annotations

import logging

from data_processing import build_feature_table, feature_columns
from model import make_predictor, make_minutes_predictor, engine_name
from validation import walk_forward_predict, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_TEST_ROUNDS = 6
MIN_TRAIN_ROUNDS = 8


def _evaluate(df, feats, predictor, test_rounds):
    return compute_metrics(walk_forward_predict(df, predictor, test_rounds, MIN_TRAIN_ROUNDS))


def main() -> None:
    table = build_feature_table("data")
    feats = feature_columns(table)
    rounds = sorted(table["round"].unique())
    test_rounds = rounds[-N_TEST_ROUNDS:]
    logger.info("Engine: %s | test rounds: %s", engine_name(), test_rounds)

    cur = _evaluate(table, feats, make_predictor(feats), test_rounds)
    new = _evaluate(table, feats, make_minutes_predictor(feats), test_rounds)

    logger.info("\n===== Walk-forward (rounds %s) =====", test_rounds)
    for name, m in (("current (2-part)", cur), ("minutes (3-band)", new)):
        logger.info("  %-18s MAE=%.4f RMSE=%.4f spearman=%.4f top10=%.3f capt=%.2f",
                    name, m["MAE"], m["RMSE"], m["spearman"], m["top10_overlap"], m["captain_return"])

    dmae = cur["MAE"] - new["MAE"]
    dspear = new["spearman"] - cur["spearman"]
    logger.info("  delta: MAE %+.4f (%+.1f%%), spearman %+.4f",
                dmae, 100 * dmae / cur["MAE"], dspear)
    if dmae > 0.01 or dspear > 0.01:
        logger.info("  VERDICT: 3-band minutes model helps. Adopt it.")
    elif dmae < -0.01:
        logger.info("  VERDICT: 3-band is worse on MAE. Keep the 2-part model.")
    else:
        logger.info("  VERDICT: within noise on MAE; check ranking before deciding.")


if __name__ == "__main__":
    main()
