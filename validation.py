"""
Leakage-free, time-aware evaluation for FPL points models.

The original pipeline used a random train/test split and a shuffled KFold on
overlapping sequence windows -- both leak the future into the past and report
optimistic, meaningless metrics. This module replaces that with:

  * walk-forward (rolling-origin) evaluation keyed on the real `round`: for each
    test round t, train only on rounds < t and predict round t;
  * mandatory naive baselines (predict-last, trailing mean, season mean) so a
    model's score is judged against a trivial heuristic, not in a vacuum;
  * ranking / captaincy metrics (Spearman, top-N overlap, captain return) on top
    of MAE/RMSE, because FPL is about ranking players, not absolute points.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# A predictor: (train_df, test_df) -> predictions aligned to test_df rows.
Predictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def walk_forward_predict(
    df: pd.DataFrame,
    predictor: Predictor,
    test_rounds: list[int],
    min_train_rounds: int = 8,
) -> pd.DataFrame:
    """Run rolling-origin prediction. Returns the test rows with a `pred` column."""
    out = []
    for t in test_rounds:
        train = df[df["round"] < t]
        if train["round"].nunique() < min_train_rounds:
            continue
        test = df[df["round"] == t].copy()
        if test.empty:
            continue
        test["pred"] = predictor(train, test)
        out.append(test)
    if not out:
        raise ValueError("No test rounds produced predictions; check test_rounds / min_train_rounds.")
    return pd.concat(out, ignore_index=True)


def _per_round_rank_metrics(preds: pd.DataFrame, top_n: int = 10) -> dict:
    """Average per-round ranking metrics."""
    spear, top_overlap, capt_return, capt_optimal = [], [], [], []
    for _, grp in preds.groupby("round"):
        if len(grp) < 5:
            continue
        a, p = grp["target_points"].to_numpy(), grp["pred"].to_numpy()
        if np.std(p) > 0 and np.std(a) > 0:
            rho, _ = spearmanr(p, a)
            if not np.isnan(rho):
                spear.append(rho)
        n = min(top_n, len(grp))
        pred_top = set(grp.nlargest(n, "pred").index)
        act_top = set(grp.nlargest(n, "target_points").index)
        top_overlap.append(len(pred_top & act_top) / n)
        # Captain: the single highest-predicted player; how many points did they score?
        capt_return.append(grp.loc[grp["pred"].idxmax(), "target_points"])
        capt_optimal.append(grp["target_points"].max())
    return {
        "spearman": float(np.mean(spear)) if spear else float("nan"),
        f"top{top_n}_overlap": float(np.mean(top_overlap)) if top_overlap else float("nan"),
        "captain_return": float(np.mean(capt_return)) if capt_return else float("nan"),
        "captain_optimal": float(np.mean(capt_optimal)) if capt_optimal else float("nan"),
    }


def compute_metrics(preds: pd.DataFrame, top_n: int = 10) -> dict:
    """Pooled error metrics + averaged per-round ranking metrics."""
    err = preds["pred"].to_numpy() - preds["target_points"].to_numpy()
    metrics = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "n_rows": int(len(preds)),
        "n_rounds": int(preds["round"].nunique()),
    }
    metrics.update(_per_round_rank_metrics(preds, top_n=top_n))
    # Captain efficiency: fraction of the best-possible captain points we captured.
    if metrics.get("captain_optimal"):
        metrics["captain_efficiency"] = metrics["captain_return"] / metrics["captain_optimal"]
    return metrics


# ---- Naive baselines (no training; just read a precomputed lagged column) ----

def _column_predictor(col: str) -> Predictor:
    def predict(_train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return test[col].fillna(0.0).to_numpy()
    return predict


BASELINES: dict[str, Predictor] = {
    "predict_last": _column_predictor("last_points"),       # last appearance's points
    "trailing_mean_5": _column_predictor("total_points_r5"),  # rolling 5-app mean
    "season_mean": _column_predictor("total_points_std"),    # expanding season mean
}


def evaluate_all(
    df: pd.DataFrame,
    model_predictor: Predictor,
    test_rounds: list[int],
    min_train_rounds: int = 8,
    extra_predictors: dict[str, Predictor] | None = None,
) -> pd.DataFrame:
    """Evaluate the model against all baselines on identical walk-forward splits."""
    predictors = {"model": model_predictor, **BASELINES}
    if extra_predictors:
        predictors.update(extra_predictors)

    rows = []
    for name, predictor in predictors.items():
        preds = walk_forward_predict(df, predictor, test_rounds, min_train_rounds)
        m = compute_metrics(preds)
        m["method"] = name
        rows.append(m)
        logger.info("Evaluated %-16s MAE=%.3f RMSE=%.3f spearman=%.3f capt=%.2f",
                    name, m["MAE"], m["RMSE"], m["spearman"], m["captain_return"])
    cols = ["method", "MAE", "RMSE", "spearman", "top10_overlap",
            "captain_return", "captain_optimal", "captain_efficiency", "n_rows", "n_rounds"]
    res = pd.DataFrame(rows)
    return res[[c for c in cols if c in res.columns]].sort_values("MAE").reset_index(drop=True)
