"""
FPL prediction + decision pipeline (rebuilt).

Stages:
  1. Build the leakage-free (player, round) feature table.
  2. Walk-forward evaluation: the hurdle GBT model vs naive baselines on
     identical rolling-origin splits (the only honest way to judge a forecaster).
  3. Train a final model on all-but-the-last round and predict the next round.
  4. Optimize a legal 15-man squad + XI + captain from those predictions.

Run:  python main.py
"""
from __future__ import annotations

import logging

import pandas as pd

import os

from data_processing import (
    build_feature_table, build_upcoming_features, build_prior_profiles,
    feature_columns, POSITION_MAP,
)
from model import FPLPointsModel, make_predictor, engine_name
from validation import evaluate_all
from optimizer import optimize_squad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fpl_prediction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

N_TEST_ROUNDS = 6      # evaluate on the most recent N rounds
MIN_TRAIN_ROUNDS = 8   # require at least this much history before scoring a round


def main() -> None:
    pd.set_option("display.width", 140)
    logger.info("Building leakage-free feature table...")
    table = build_feature_table("data")
    feats = feature_columns(table)
    logger.info("Model engine: %s | %d features", engine_name(), len(feats))

    max_round = int(table["round"].max())
    test_rounds = list(range(max_round - N_TEST_ROUNDS + 1, max_round + 1))

    logger.info("Walk-forward evaluation on rounds %s (model vs baselines)...", test_rounds)
    results = evaluate_all(
        table,
        model_predictor=make_predictor(feats),
        test_rounds=test_rounds,
        min_train_rounds=MIN_TRAIN_ROUNDS,
    )
    logger.info("\n===== Walk-forward results (lower MAE/RMSE better; higher spearman/captain better) =====\n%s",
                results.to_string(index=False))

    model_mae = results.loc[results["method"] == "model", "MAE"].iloc[0]
    best_baseline = results[results["method"] != "model"].sort_values("MAE").iloc[0]
    verdict = "BEATS" if model_mae < best_baseline["MAE"] else "DOES NOT beat"
    logger.info("Model %s the best baseline (%s): MAE %.3f vs %.3f",
                verdict, best_baseline["method"], model_mae, best_baseline["MAE"])

    # ---- FORECAST the next, NOT-YET-PLAYED gameweek (the real use case) ----
    # Train on ALL available data, then predict round (max_round + 1) using each
    # player's form so far + that round's known fixtures. There are no actuals
    # for this round -- it is a genuine forecast, not a re-scored past gameweek.
    forecast_round = max_round + 1
    logger.info("Training final model on all data and FORECASTING round %d (no actuals)...",
                forecast_round)
    model = FPLPointsModel().fit(table, feats)

    # Cold start: at the start of a NEW season there is no within-season form, so
    # seed early gameweeks from the previous season. Point FPL_PRIOR_SEASON_DIR at
    # last season's ingested data to enable it; mid-season it is unnecessary.
    prior_dir = os.environ.get("FPL_PRIOR_SEASON_DIR")
    profiles = None
    if prior_dir:
        logger.info("Cold-start enabled: seeding from prior season at %s", prior_dir)
        profiles = build_prior_profiles(prior_dir, current_base_dir="data")

    upcoming = build_upcoming_features("data", target_round=forecast_round,
                                       prior_profiles=profiles)
    upcoming["pred"] = model.predict(upcoming)

    top = upcoming.sort_values("pred", ascending=False).head(15)
    logger.info("\nTop 15 FORECAST players for round %d (upcoming, unplayed):\n%s", forecast_round,
                top.assign(pos=top["element_type"].map(POSITION_MAP))
                   [["name", "pos", "value", "pred"]].to_string(index=False))

    # ---- Optimize squad on the forecast ----
    logger.info("Optimizing squad under FPL constraints for round %d...", forecast_round)
    players = upcoming.rename(columns={"element_type": "pos"})
    players["price"] = players["value"] / 10.0
    sol = optimize_squad(players[["player_id", "name", "pos", "team", "price", "pred"]])
    squad = sol.squad.assign(pos=lambda d: d["pos"].map(POSITION_MAP))
    logger.info("\n===== Optimal squad (£%.1fm, XI expected %.1f pts) =====\n%s",
                sol.total_cost, sol.xi_expected_points,
                squad[["name", "pos", "team", "price", "pred", "in_xi", "is_captain"]]
                .to_string(index=False))
    captain = squad[squad["is_captain"]].iloc[0]
    logger.info("Captain: %s (%.1f predicted pts, doubled)", captain["name"], captain["pred"])


if __name__ == "__main__":
    main()
