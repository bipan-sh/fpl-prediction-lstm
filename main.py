"""
FPL prediction + decision pipeline (rebuilt), season-aware.

What it does, depending on where the season is:
  * Mid-season (enough played gameweeks): walk-forward evaluation vs baselines,
    then train on all data and FORECAST the next unplayed gameweek.
  * New-season opener (little/no current-season data yet): skip evaluation (nothing
    to score), train on LAST season, and forecast GW1 via the cold-start profiles.

The forecast target is always the next UNFINISHED gameweek read from fixtures, so
it self-adjusts as the real season progresses.

Run:
  python data_ingestion.py                      # pull current-season data (needs internet)
  python main.py                                 # mid-season
  FPL_PRIOR_SEASON_DIR=data_2025_26 python main.py   # new-season opener (train+seed from last year)
  FPL_INGEST=1 python main.py                    # ingest fresh data first, then run
"""
from __future__ import annotations

import os
import logging

import pandas as pd

from data_ingestion import ingest_data
from data_processing import (
    build_feature_table, build_upcoming_features, build_prior_profiles,
    feature_columns, next_unfinished_round, POSITION_MAP,
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
DATA_DIR = os.environ.get("FPL_DATA_DIR", "data")


def _evaluate(table, feats) -> None:
    max_round = int(table["round"].max())
    test_rounds = list(range(max_round - N_TEST_ROUNDS + 1, max_round + 1))
    logger.info("Walk-forward evaluation on rounds %s (model vs baselines)...", test_rounds)
    results = evaluate_all(table, model_predictor=make_predictor(feats),
                           test_rounds=test_rounds, min_train_rounds=MIN_TRAIN_ROUNDS)
    logger.info("\n===== Walk-forward results (lower MAE/RMSE better) =====\n%s",
                results.to_string(index=False))
    model_mae = results.loc[results["method"] == "model", "MAE"].iloc[0]
    best = results[results["method"] != "model"].sort_values("MAE").iloc[0]
    logger.info("Model %s the best baseline (%s): MAE %.3f vs %.3f",
                "BEATS" if model_mae < best["MAE"] else "DOES NOT beat",
                best["method"], model_mae, best["MAE"])


def main() -> None:
    pd.set_option("display.width", 140)

    if os.environ.get("FPL_INGEST") == "1":
        logger.info("FPL_INGEST=1 -> pulling current-season data from the FPL API...")
        ingest_data(DATA_DIR)

    table = build_feature_table(DATA_DIR)
    feats = feature_columns(table)
    played_rounds = sorted(table["round"].unique())
    forecast_round = next_unfinished_round(DATA_DIR)
    prior_dir = os.environ.get("FPL_PRIOR_SEASON_DIR")
    logger.info("Engine: %s | %d features | %d played rounds | next unfinished GW: %s",
                engine_name(), len(feats), len(played_rounds), forecast_round)

    if forecast_round is None:
        logger.warning("All fixtures are finished — the season is over. Nothing to forecast.")
        return

    opener = len(played_rounds) < MIN_TRAIN_ROUNDS + 2  # too little current-season data to validate

    if not opener:
        # ---- Mid-season: validate, then train on everything available ----
        _evaluate(table, feats)
        logger.info("Training final model on all data; FORECASTING round %d (no actuals)...",
                    forecast_round)
        model = FPLPointsModel().fit(table, feats)
        profiles = build_prior_profiles(prior_dir, DATA_DIR) if prior_dir else None
    else:
        # ---- New-season opener: no within-season history to learn/validate on ----
        logger.info("SEASON-OPENER mode: only %d played round(s) — skipping walk-forward "
                    "(nothing to validate yet).", len(played_rounds))
        if not prior_dir:
            logger.error("Opener needs last season's data to train + seed. Re-run with "
                         "FPL_PRIOR_SEASON_DIR=<last-season-dir>. Aborting.")
            return
        logger.info("Training on prior season (%s) and seeding GW%d via cold-start.",
                    prior_dir, forecast_round)
        prior_table = build_feature_table(prior_dir)
        model = FPLPointsModel().fit(prior_table, feature_columns(prior_table))
        profiles = build_prior_profiles(prior_dir, DATA_DIR)

    # ---- Forecast the next unplayed gameweek + optimize the squad ----
    upcoming = build_upcoming_features(DATA_DIR, target_round=forecast_round, prior_profiles=profiles)
    upcoming["raw_pred"] = model.predict(upcoming)
    # Downweight injured/suspended/doubtful players using current availability
    # (a live signal known before kickoff; illustrative only on an offline snapshot).
    upcoming["pred"] = upcoming["raw_pred"] * upcoming["availability"]
    logger.info("Applied availability downweight to %d flagged players.",
                int((upcoming["availability"] < 1.0).sum()))

    top = upcoming.sort_values("pred", ascending=False).head(15)
    logger.info("\nTop 15 FORECAST players for round %d (upcoming, unplayed):\n%s", forecast_round,
                top.assign(pos=top["element_type"].map(POSITION_MAP))
                   [["name", "pos", "value", "pred"]].to_string(index=False))

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
