"""
Squad optimization for FPL via integer linear programming (scipy.optimize.milp,
HiGHS solver -- the same solver the leading FPL optimization tools use).

Turns per-player expected points into an actual, legal FPL decision: pick the
15-man squad, the starting XI, and the captain that maximize expected points
subject to every real FPL rule:

  * squad of 15: exactly 2 GK, 5 DEF, 5 MID, 3 FWD
  * budget <= 100.0m
  * at most 3 players per club
  * starting XI of 11 with a legal formation
      GK = 1, DEF in [3,5], MID in [2,5], FWD in [1,3]
  * captain is one of the starting XI (counts double)

The original repo had NO optimizer at all -- it only printed predictions.
This is the foundation; multi-gameweek transfer planning and chip timing
(Wildcard / Free Hit / Bench Boost / Triple Captain) build on the same model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

logger = logging.getLogger(__name__)

# element_type -> required squad count and starting-XI bounds
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


@dataclass
class SquadSolution:
    squad: pd.DataFrame          # 15 players, with `in_xi` and `is_captain` flags
    total_cost: float
    xi_expected_points: float    # starting XI + captain bonus
    status: str


def optimize_squad(
    players: pd.DataFrame,
    budget: float = 100.0,
    pred_col: str = "pred",
    force_in: list | None = None,
    exclude: list | None = None,
) -> SquadSolution:
    """Select the optimal 15-man squad, XI and captain.

    `players` must have columns: player_id, name, pos (1-4), team, price, <pred_col>.
    `force_in` player_ids are pinned into the squad; `exclude` player_ids are dropped
    (used by the web UI's lock / ban controls). `budget` is in millions.
    """
    df = players.dropna(subset=["pos", "team", "price", pred_col]).copy()
    df = df[df["pos"].isin(SQUAD_QUOTA)].copy()
    if exclude:
        df = df[~df["player_id"].isin(exclude)]
    df = df.reset_index(drop=True)
    n = len(df)
    if n < 15:
        raise ValueError(f"Need >=15 eligible players, got {n}")

    pos = df["pos"].to_numpy()
    price = df["price"].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    teams = df["team"].to_numpy()

    # Decision vars: squad[0:n], xi[n:2n], captain[2n:3n]  (all binary)
    N = 3 * n
    SQ, XI, CAP = 0, n, 2 * n

    # Objective: maximize XI points + captain bonus -> minimize the negative.
    c = np.zeros(N)
    c[XI:XI + n] = -pred
    c[CAP:CAP + n] = -pred

    rows, lbs, ubs = [], [], []

    def add(coeffs: dict[int, float], lb: float, ub: float):
        r = np.zeros(N)
        for idx, v in coeffs.items():
            r[idx] = v
        rows.append(r)
        lbs.append(lb)
        ubs.append(ub)

    # Squad size and position quotas.
    add({SQ + i: 1 for i in range(n)}, 15, 15)
    for p, q in SQUAD_QUOTA.items():
        add({SQ + i: 1 for i in range(n) if pos[i] == p}, q, q)

    # Budget.
    add({SQ + i: price[i] for i in range(n)}, 0, budget)

    # Max 3 per club.
    for tm in np.unique(teams):
        add({SQ + i: 1 for i in range(n) if teams[i] == tm}, 0, 3)

    # Pin user-locked players into the squad.
    if force_in:
        ids = df["player_id"].to_numpy()
        for pid in force_in:
            idx = np.where(ids == pid)[0]
            if len(idx):
                add({SQ + int(idx[0]): 1}, 1, 1)

    # Starting XI: exactly 11, must be in squad, legal formation.
    add({XI + i: 1 for i in range(n)}, 11, 11)
    for i in range(n):  # xi_i <= squad_i
        add({XI + i: 1, SQ + i: -1}, -np.inf, 0)
    for p, (lo, hi) in XI_BOUNDS.items():
        add({XI + i: 1 for i in range(n) if pos[i] == p}, lo, hi)

    # Captain: exactly one, must be a starter.
    add({CAP + i: 1 for i in range(n)}, 1, 1)
    for i in range(n):  # captain_i <= xi_i
        add({CAP + i: 1, XI + i: -1}, -np.inf, 0)

    A = np.array(rows)
    constraints = LinearConstraint(A, np.array(lbs), np.array(ubs))
    res = milp(
        c,
        integrality=np.ones(N),
        bounds=Bounds(np.zeros(N), np.ones(N)),
        constraints=constraints,
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x = np.round(res.x).astype(int)
    in_squad = x[SQ:SQ + n] == 1
    sol = df[in_squad].copy()
    sol["in_xi"] = x[XI:XI + n][in_squad] == 1
    sol["is_captain"] = x[CAP:CAP + n][in_squad] == 1
    sol = sol.sort_values(["in_xi", "pos", pred_col], ascending=[False, True, False])

    xi_points = float((sol.loc[sol["in_xi"], pred_col]).sum()
                      + sol.loc[sol["is_captain"], pred_col].sum())
    return SquadSolution(
        squad=sol,
        total_cost=float(sol["price"].sum()),
        xi_expected_points=xi_points,
        status=res.message,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from data_processing import build_feature_table, feature_columns, POSITION_MAP
    from model import FPLPointsModel

    table = build_feature_table()
    feats = feature_columns(table)
    target_round = int(table["round"].max())
    train = table[table["round"] < target_round]
    test = table[table["round"] == target_round].copy()
    test["pred"] = FPLPointsModel().fit(train, feats).predict(test)

    players = test.rename(columns={"element_type": "pos", "value": "price_raw"})
    players["price"] = players["price_raw"] / 10.0
    sol = optimize_squad(players[["player_id", "name", "pos", "team", "price", "pred"]])
    print(f"\nOptimal squad — cost £{sol.total_cost:.1f}m, "
          f"XI expected points {sol.xi_expected_points:.1f}")
    show = sol.squad.copy()
    show["pos"] = show["pos"].map(POSITION_MAP)
    print(show[["name", "pos", "team", "price", "pred", "in_xi", "is_captain"]].to_string(index=False))
