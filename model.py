"""
Position-specific, hurdle-structured gradient-boosted model for FPL points.

Why this replaces the LSTM
--------------------------
The strongest *free-data* public model, OpenFPL (2025), matches the paid
FPL Review projections using position-specific ensembles of gradient-boosted
trees -- not deep sequence models. Trees also sidestep two problems the old
LSTM had: they need no feature scaling (so no scaler-leakage), and they handle
missing values natively (so a player's early-season NaN form features are fine).

Hurdle structure (handles the ~50%-zero, right-skewed target)
-------------------------------------------------------------
For each position we fit TWO models:
  1. appear classifier:  P(player features for >= 1 minute)   [all rows]
  2. points regressor:   E[points | the player appeared]      [appeared rows]
The expected points is  P(appear) * E[points | appeared], clipped at 0.
Modelling "will they play" separately is the single biggest accuracy lever in
FPL, because minutes dominate the points distribution.

Engine: LightGBM if installed, else sklearn HistGradientBoosting (same
gradient-boosted-trees family; ships with scikit-learn, no extra install).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical, engine-agnostic hyperparameters. Selected by nested walk-forward
# tuning (see tune.py): on a locked final-test block these improved held-out MAE
# ~2.4% (1.071 -> 1.046) plus RMSE and rank vs. the initial hand-set values
# {lr 0.05, leaves 31, trees 300, min_leaf 20, l2 1.0}. The winner is a simpler,
# more-regularized model, which is the right bias for a noisy target.
DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "max_leaves": 15,
    "n_trees": 200,
    "min_leaf": 100,
    "l2": 0.0,
}

try:  # optional faster engine
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

    _ENGINE = "lightgbm"

    def _estimator(kind: str, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        # subsample needs subsample_freq>=1 to actually bag; seed for repeatability.
        common = dict(n_estimators=int(p["n_trees"]), learning_rate=p["learning_rate"],
                      num_leaves=int(p["max_leaves"]), min_child_samples=int(p["min_leaf"]),
                      reg_lambda=p["l2"], subsample=0.8, subsample_freq=1,
                      random_state=42, verbosity=-1)
        return LGBMRegressor(**common) if kind == "reg" else LGBMClassifier(**common)
except Exception:  # pragma: no cover - fallback path
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    )

    _ENGINE = "hist_gradient_boosting"

    def _estimator(kind: str, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        common = dict(max_iter=int(p["n_trees"]), learning_rate=p["learning_rate"],
                      max_leaf_nodes=int(p["max_leaves"]), min_samples_leaf=int(p["min_leaf"]),
                      l2_regularization=p["l2"], random_state=42)
        return (HistGradientBoostingRegressor(**common) if kind == "reg"
                else HistGradientBoostingClassifier(**common))


def _make_regressor(params=None):
    return _estimator("reg", params)


def _make_classifier(params=None):
    return _estimator("clf", params)


class FPLPointsModel:
    """Position-specific hurdle model: P(appear) x E[points | appeared]."""

    def __init__(self, positions=(1, 2, 3, 4), params: dict | None = None):
        self.positions = positions
        self.params = params  # canonical hyperparameters; None -> DEFAULT_PARAMS
        self.appear_models: dict[int, object] = {}
        self.points_models: dict[int, object] = {}
        self.fallback_rate: dict[int, float] = {}
        self.feature_cols: list[str] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "FPLPointsModel":
        self.feature_cols = feature_cols
        for pos in self.positions:
            sub = train[train["element_type"] == pos]
            if len(sub) < 50:
                continue
            X = sub[feature_cols]
            y_appear = sub["target_appeared"].to_numpy()
            self.fallback_rate[pos] = float(y_appear.mean())
            # Appearance classifier (needs both classes present).
            if len(np.unique(y_appear)) == 2:
                clf = _make_classifier(self.params)
                clf.fit(X, y_appear)
                self.appear_models[pos] = clf
            # Conditional points regressor (only on appearances).
            appeared = sub[sub["target_appeared"] == 1]
            if len(appeared) >= 30:
                reg = _make_regressor(self.params)
                reg.fit(appeared[feature_cols], appeared["target_points"].to_numpy())
                self.points_models[pos] = reg
        return self

    def _predict_pos(self, pos: int, X: pd.DataFrame) -> np.ndarray:
        reg = self.points_models.get(pos)
        if reg is None:
            return np.zeros(len(X))
        cond = np.clip(reg.predict(X), 0, None)
        clf = self.appear_models.get(pos)
        if clf is not None:
            p_appear = clf.predict_proba(X)[:, 1]
        else:
            p_appear = np.full(len(X), self.fallback_rate.get(pos, 1.0))
        return p_appear * cond

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for pos in self.positions:
            mask = (df["element_type"] == pos).to_numpy()
            if mask.any():
                preds[mask] = self._predict_pos(pos, df.loc[mask, self.feature_cols])
        return preds


def make_predictor(feature_cols: list[str], params: dict | None = None
                   ) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Adapter for validation.walk_forward_predict: refits fresh on each fold."""
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = FPLPointsModel(params=params).fit(train, feature_cols)
        return model.predict(test)
    return predict


# Minutes bands: how long a player is on the pitch drives appearance points
# (0 / 1 / 2) and clean-sheet eligibility (needs 60+), so we model the band, not
# just "did they appear".
def _minutes_band(minutes) -> np.ndarray:
    m = np.asarray(minutes)
    return np.where(m == 0, 0, np.where(m < 60, 1, 2))  # 0=DNP, 1=cameo, 2=start


class MinutesHurdleModel:
    """Finer hurdle: P(cameo)*E[pts|cameo] + P(start)*E[pts|start], per position.

    A 3-class minutes model (DNP / cameo 1-59' / start 60'+) replaces the binary
    "appears?" classifier, and points are predicted separately for cameo vs start
    appearances (a cameo has a far lower points ceiling than a full start). Bands
    too sparse for their own regressor fall back to a pooled appeared-rows model.
    """

    def __init__(self, positions=(1, 2, 3, 4), params: dict | None = None):
        self.positions = positions
        self.params = params
        self.band_clf: dict[int, object] = {}
        self.reg_cameo: dict[int, object] = {}
        self.reg_start: dict[int, object] = {}
        self.reg_pool: dict[int, object] = {}
        self.appear_rate: dict[int, float] = {}
        self.feature_cols: list[str] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> "MinutesHurdleModel":
        self.feature_cols = feature_cols
        for pos in self.positions:
            sub = train[train["element_type"] == pos]
            if len(sub) < 50:
                continue
            X, y_pts = sub[feature_cols], sub["target_points"].to_numpy()
            band = _minutes_band(sub["minutes"])
            self.appear_rate[pos] = float((band >= 1).mean())
            if len(np.unique(band)) >= 2:
                clf = _make_classifier(self.params)
                clf.fit(X, band)
                self.band_clf[pos] = clf
            appeared = band >= 1
            if appeared.sum() >= 30:
                r = _make_regressor(self.params)
                r.fit(sub.loc[appeared, feature_cols], y_pts[appeared])
                self.reg_pool[pos] = r
            for b, store in ((1, self.reg_cameo), (2, self.reg_start)):
                rows = band == b
                if rows.sum() >= 40:
                    r = _make_regressor(self.params)
                    r.fit(sub.loc[rows, feature_cols], y_pts[rows])
                    store[pos] = r
        return self

    def _predict_pos(self, pos: int, X: pd.DataFrame) -> np.ndarray:
        pool = self.reg_pool.get(pos)
        if pool is None:
            return np.zeros(len(X))
        e_pool = np.clip(pool.predict(X), 0, None)
        e_cameo = np.clip(self.reg_cameo[pos].predict(X), 0, None) if pos in self.reg_cameo else e_pool
        e_start = np.clip(self.reg_start[pos].predict(X), 0, None) if pos in self.reg_start else e_pool
        clf = self.band_clf.get(pos)
        if clf is None:  # degenerate: fall back to appear-rate * pooled
            return self.appear_rate.get(pos, 1.0) * e_pool
        proba = clf.predict_proba(X)
        cls = list(clf.classes_)

        def pcol(b):
            return proba[:, cls.index(b)] if b in cls else np.zeros(len(X))

        return pcol(1) * e_cameo + pcol(2) * e_start

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for pos in self.positions:
            mask = (df["element_type"] == pos).to_numpy()
            if mask.any():
                preds[mask] = self._predict_pos(pos, df.loc[mask, self.feature_cols])
        return preds


def make_minutes_predictor(feature_cols: list[str], params: dict | None = None
                           ) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Adapter for walk_forward_predict using the 3-band minutes hurdle."""
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return MinutesHurdleModel(params=params).fit(train, feature_cols).predict(test)
    return predict


def engine_name() -> str:
    return _ENGINE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from data_processing import build_feature_table, feature_columns

    table = build_feature_table()
    feats = feature_columns(table)
    train = table[table["round"] < 24]
    test = table[table["round"] == 24].copy()
    model = FPLPointsModel().fit(train, feats)
    test["pred"] = model.predict(test)
    print(f"Engine: {_ENGINE}")
    print(test.sort_values("pred", ascending=False)
          [["name", "element_type", "pred", "target_points"]].head(10).to_string())
