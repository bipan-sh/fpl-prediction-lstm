"""
Lightweight smoke + correctness tests for the rebuilt FPL pipeline.

Runs with plain Python (no pytest needed):   python3 tests/test_pipeline.py
Or, if you have pytest:                       pytest -q

Checks:
  1. The feature table builds and is leakage-shaped (no future info in features).
  2. The model produces signal (positive rank correlation with actuals).
  3. The optimizer returns a LEGAL FPL squad (all constraints satisfied).
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

# allow running from repo root or tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import (
    build_feature_table, build_upcoming_features, build_prior_profiles, feature_columns,
    next_unfinished_round,
)
from model import FPLPointsModel
from optimizer import optimize_squad, SQUAD_QUOTA, XI_BOUNDS


def _load():
    table = build_feature_table("data")
    feats = feature_columns(table)
    return table, feats


def test_feature_table_is_leakage_free():
    table, feats = _load()
    # one row per (player, round)
    assert not table.duplicated(["player_id", "round"]).any(), "duplicate (player,round) rows"
    # uses the real round axis, not a fabricated 1..N gameweek
    assert table["round"].max() > 1
    # a player's FIRST appearance must have NO prior-form features (pure lag => NaN)
    first_rows = table.sort_values("round").groupby("player_id").head(1)
    assert first_rows["last_points"].isna().all(), "lagged feature leaks into first appearance"
    # targets present and sane
    assert {"target_points", "target_appeared"}.issubset(table.columns)
    assert table["target_appeared"].isin([0, 1]).all()
    print(f"  [ok] feature table: {len(table)} rows, {len(feats)} features, leakage-shaped")


def test_model_has_predictive_signal():
    table, feats = _load()
    t = int(table["round"].max())
    train = table[table["round"] < t]
    test = table[table["round"] == t].copy()
    test["pred"] = FPLPointsModel().fit(train, feats).predict(test)
    rho, _ = spearmanr(test["pred"], test["target_points"])
    assert rho > 0.3, f"weak rank correlation rho={rho:.3f}"
    assert (test["pred"] >= 0).all(), "negative predicted points"
    print(f"  [ok] model signal: round {t} Spearman={rho:.3f}, preds non-negative")


def test_optimizer_returns_legal_squad():
    table, feats = _load()
    t = int(table["round"].max())
    train = table[table["round"] < t]
    test = table[table["round"] == t].copy()
    test["pred"] = FPLPointsModel().fit(train, feats).predict(test)
    players = test.rename(columns={"element_type": "pos"})
    players["price"] = players["value"] / 10.0
    sol = optimize_squad(players[["player_id", "name", "pos", "team", "price", "pred"]])
    sq = sol.squad

    assert len(sq) == 15, "squad must be 15 players"
    for pos, q in SQUAD_QUOTA.items():
        assert (sq["pos"] == pos).sum() == q, f"position {pos} count != {q}"
    assert sol.total_cost <= 100.0 + 1e-6, f"over budget: {sol.total_cost}"
    assert (sq.groupby("team").size() <= 3).all(), "more than 3 from one club"
    assert sq["in_xi"].sum() == 11, "starting XI must be 11"
    for pos, (lo, hi) in XI_BOUNDS.items():
        c = ((sq["pos"] == pos) & sq["in_xi"]).sum()
        assert lo <= c <= hi, f"illegal formation for pos {pos}: {c}"
    assert sq["is_captain"].sum() == 1, "exactly one captain"
    cap = sq[sq["is_captain"]].iloc[0]
    assert cap["in_xi"], "captain not in starting XI"
    print(f"  [ok] optimizer: legal squad £{sol.total_cost:.1f}m, captain {cap['name']}")


def test_cold_start_seeds_gw1():
    # GW1 (round 1) has zero within-season history, so unseeded form is all-NaN.
    raw = build_upcoming_features("data", target_round=1)
    assert raw["total_points_r5"].isna().all(), "expected no within-season form at GW1"

    # With a prior-season profile, GW1 form is seeded (non-NaN) and differentiates players.
    profiles = build_prior_profiles("data")
    seeded = build_upcoming_features("data", target_round=1, prior_profiles=profiles)
    cov = seeded["total_points_r5"].notna().mean()
    assert cov > 0.9, f"cold-start should seed almost all players, got {cov:.2f}"
    assert seeded["total_points_r5"].std() > 0, "seeded form has no variation"
    # At GW1, games_played==0 so seeded form must equal the prior per-game value.
    assert (seeded["games_played"] == 0).all()
    print(f"  [ok] cold-start: GW1 unseeded NaN -> seeded {cov*100:.0f}% players with prior form")


def test_availability_multiplier():
    up = build_upcoming_features("data", target_round=27)
    assert "availability" in up.columns
    a = up["availability"]
    assert ((a >= 0) & (a <= 1)).all(), "availability must be in [0, 1]"
    assert (a < 1.0).any(), "expected some flagged (injured/suspended/doubtful) players"
    print(f"  [ok] availability: range [{a.min():.2f}, {a.max():.2f}], {(a < 1).sum()} flagged")


def test_next_unfinished_round():
    # The forecast target is the next unfinished GW from fixtures. The bundled
    # 2024-25 archive has match data through round 26, so the next GW is 27.
    nxt = next_unfinished_round("data")
    assert nxt == 27, f"expected next unfinished round 27 on the archive, got {nxt}"
    print(f"  [ok] next unfinished round = {nxt} (forecast target)")


if __name__ == "__main__":
    tests = [
        test_feature_table_is_leakage_free,
        test_model_has_predictive_signal,
        test_optimizer_returns_legal_squad,
        test_cold_start_seeds_gw1,
        test_availability_multiplier,
        test_next_unfinished_round,
    ]
    failed = 0
    for fn in tests:
        print(f"- {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
