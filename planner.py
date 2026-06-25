"""
Multi-gameweek planning: project points over a horizon, suggest transfers, hint chips.

Projection method (standard for FPL planners): hold each player's *current* form
fixed and vary only the fixture (opponent strength, home/away) for each future
gameweek. So gw+k's projection = the model applied to today's form + gw+k's known
fixture. Summed over the horizon this ranks who has the best run of games.

Transfers are then optimised from a current squad: pick the squad reachable within
`max_transfers` changes that maximises horizon points, minus 4 per paid transfer
(beyond the free ones). Chip hints are simple heuristics over the projection.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_processing import build_upcoming_features, next_unfinished_round, POSITION_MAP
from optimizer import optimize_squad

logger = logging.getLogger(__name__)


def horizon_projections(model, base_dir: str = "data", start_round: int | None = None,
                        horizon: int = 5, profiles=None, fixtures_max: int = 38) -> pd.DataFrame:
    """One row per player with predicted points for each of the next `horizon` GWs.

    Returns columns: player_id, name, pos, team, price, gw<r>... , horizon (sum).
    """
    if start_round is None:
        start_round = next_unfinished_round(base_dir) or 1
    rounds = [r for r in range(start_round, start_round + horizon) if r <= fixtures_max]
    gw_cols, gw_frames, meta = [], [], None
    for r in rounds:
        up = build_upcoming_features(base_dir, target_round=r, prior_profiles=profiles)
        if up.empty:
            continue
        col = f"gw{r}"
        # predict BEFORE renaming (the model reads the 'element_type' column)
        up[col] = np.clip(model.predict(up) * up["availability"], 0, None).round(2)
        up = up.rename(columns={"element_type": "pos"})
        up["price"] = up["value"] / 10.0
        gw_cols.append(col)
        gw_frames.append(up[["player_id", col]])
        # accumulate static metadata across ALL rounds so a player whose first
        # projected GW is a blank still keeps name/pos/team/price (OPT-1).
        m = up[["player_id", "name", "pos", "team", "price"]]
        meta = m if meta is None else pd.concat([meta, m]).drop_duplicates("player_id")
    if not gw_frames:
        raise ValueError("No upcoming fixtures to project.")
    preds = gw_frames[0]
    for f in gw_frames[1:]:
        preds = preds.merge(f, on="player_id", how="outer")
    base = meta.merge(preds, on="player_id", how="right")
    base[gw_cols] = base[gw_cols].fillna(0.0)
    base["horizon"] = base[gw_cols].sum(axis=1).round(2)
    logger.info("Projected %d players over GWs %s", len(base), rounds)
    return base.sort_values("horizon", ascending=False).reset_index(drop=True)


def suggest_transfers(proj: pd.DataFrame, current_ids: list, max_transfers: int = 2,
                      free_transfers: int = 1, bank: float = 0.0) -> dict:
    """Best transfers from `current_ids` for the projected horizon (vs holding)."""
    cols = ["player_id", "name", "pos", "team", "price", "horizon"]
    p = proj[cols].copy()
    pool = set(p["player_id"])
    # Current players with no fixture in the horizon aren't selectable -> they are
    # forced sells; don't let that crash the 'keep' baseline (OPT-2).
    present = [int(i) for i in current_ids if i in pool]
    forced = 15 - len(present)
    # Players not in the pool are forced sells; we don't have their price here, so
    # credit their sale value at the pool mean so the replacement is affordable.
    budget = float(p[p["player_id"].isin(present)]["price"].sum()) \
        + forced * float(p["price"].mean()) + bank

    keep = optimize_squad(p, budget=budget, pred_col="horizon", current_ids=present,
                          max_transfers=forced, free_transfers=free_transfers)
    best = optimize_squad(p, budget=budget, pred_col="horizon", current_ids=present,
                          max_transfers=forced + max_transfers, free_transfers=free_transfers)
    nm = dict(zip(p["player_id"], p["name"]))
    net = (best.xi_expected_points - 4 * best.hits) - (keep.xi_expected_points - 4 * keep.hits)
    return {
        "keep_points": round(keep.xi_expected_points, 1),
        "best_points": round(best.xi_expected_points, 1),
        "hits": best.hits,
        "net_gain": round(net, 1),
        "transfers_in": [{"id": i, "name": nm.get(i, i)} for i in (best.transfers_in or [])],
        "transfers_out": [{"id": i, "name": nm.get(i, i)} for i in (best.transfers_out or [])],
        "horizon_squad": best,
    }


def chip_hints(proj: pd.DataFrame, squad_ids: list, gw_cols: list) -> dict:
    """Cheap heuristics: which GW looks best for Triple Captain / Bench Boost."""
    sq = proj[proj["player_id"].isin(squad_ids)]
    tc = {c: float(sq[c].max()) for c in gw_cols}                 # best single captain pick
    bb = {c: float(sq[c].nsmallest(4).sum()) for c in gw_cols}    # weakest 4 ~ bench floor
    tc_gw = max(tc, key=tc.get)
    bb_gw = max(bb, key=bb.get)
    return {
        "tripleCaptain": {"gw": tc_gw, "value": round(tc[tc_gw], 1)},
        "benchBoost": {"gw": bb_gw, "value": round(bb[bb_gw], 1)},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from data_processing import build_feature_table, feature_columns
    from model import FPLPointsModel

    table = build_feature_table("data")
    feats = feature_columns(table)
    model = FPLPointsModel().fit(table, feats)
    proj = horizon_projections(model, "data", horizon=5)
    gw_cols = [c for c in proj.columns if c.startswith("gw")]
    print("\nTop 10 over the next", len(gw_cols), "GWs:")
    show = proj.head(10).assign(pos=lambda d: d["pos"].map(POSITION_MAP))
    print(show[["name", "pos", "price"] + gw_cols + ["horizon"]].to_string(index=False))

    # demo: transfers from the single-GW optimal squad
    from optimizer import optimize_squad
    g1 = proj.rename(columns={gw_cols[0]: "pred"})
    cur = optimize_squad(g1[["player_id", "name", "pos", "team", "price", "pred"]]).squad["player_id"].tolist()
    sug = suggest_transfers(proj, cur, max_transfers=2, free_transfers=1)
    print(f"\nTransfer plan (≤2): net {sug['net_gain']:+.1f} pts over horizon, {sug['hits']} hit(s)")
    print("  IN :", [t["name"] for t in sug["transfers_in"]])
    print("  OUT:", [t["name"] for t in sug["transfers_out"]])
    print("Chip hints:", chip_hints(proj, cur, gw_cols))
