"""
Leakage-free feature engineering for FPL next-gameweek points prediction.

Rebuild notes (vs. the original LSTM pipeline):
  * Uses the REAL `round` column as the time axis (the old code fabricated a
    `gameweek` from row order, which broke for late joiners, double and blank
    gameweeks).
  * Aggregates to ONE row per (player, round) so double gameweeks are handled
    explicitly instead of being silently split into separate "gameweeks".
  * Every predictive feature is a rolling / lagged statistic SHIFTED by one
    appearance, so the target round's own outcomes never leak into its features.
  * No global StandardScaler. Gradient-boosted trees are scale invariant, which
    also removes the "fit scaler on the whole dataset before the split" leak.
  * Builds a rich feature set (xG/xA, ICT, BPS, form, fixture/opponent strength,
    home/away) instead of just [minutes, goals, assists].

The output is a tidy DataFrame, one row per (player_id, round), with:
  - identity:   player_id, name, round, team, element_type (1=GK 2=DEF 3=MID 4=FWD)
  - target:     target_points (that round's total_points), target_appeared (min>=1)
  - features:   FEATURE_COLS (all known strictly before the round kicks off)
Managers (element_type 5, introduced 2024-25) are excluded.
"""
from __future__ import annotations

import os
import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MNG"}
PLAYER_POSITIONS = (1, 2, 3, 4)  # exclude managers (5)

# Per-appearance stats we turn into lagged/rolling "form" features.
_FORM_STATS = [
    "total_points", "minutes", "starts",
    "goals_scored", "assists",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "ict_index", "influence", "creativity", "threat",
    "bps", "bonus", "saves", "goals_conceded", "clean_sheets",
]

# Rolling window lengths (in prior appearances).
_WINDOWS = (3, 5)


def _read_players_raw(base_dir: str) -> Optional[pd.DataFrame]:
    """Static per-player attributes: position, club, current price.

    NOTE: the original code looked for ``playerraw.csv`` while ingestion saved
    ``players_raw.csv`` -- so positions never loaded. Fixed here.
    """
    path = os.path.join(base_dir, "players_raw.csv")
    if not os.path.exists(path):
        logger.warning("players_raw.csv not found at %s; positions/prices unavailable.", path)
        return None
    return pd.read_csv(path)


def _read_teams(base_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(base_dir, "teams.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_player_gw(base_dir: str) -> pd.DataFrame:
    """Concatenate every per-player gw.csv into one long frame."""
    players_dir = os.path.join(base_dir, "players")
    files = sorted(glob.glob(os.path.join(players_dir, "*", "gw.csv")))
    if not files:
        raise FileNotFoundError(f"No player gw.csv files under {players_dir}")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:  # a corrupt file should not kill the whole run
            logger.warning("Skipping unreadable %s: %s", f, exc)
    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d player-fixture rows from %d files", len(df), len(files))
    return df


def _aggregate_to_player_round(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (player_id, round).

    Double gameweeks (two fixtures in one round) are summed for counting stats,
    so a player's DGW is a single, correctly-weighted observation rather than
    two fake consecutive "gameweeks".
    """
    if "round" not in df.columns:
        raise KeyError("Expected a 'round' column in the gameweek data.")

    sum_cols = [
        "minutes", "total_points", "goals_scored", "assists",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "ict_index", "influence", "creativity",
        "threat", "bps", "bonus", "saves", "goals_conceded", "starts",
        "own_goals", "yellow_cards", "red_cards", "penalties_missed",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]

    spec = {c: "sum" for c in sum_cols}
    if "clean_sheets" in df.columns:
        spec["clean_sheets"] = "max"
    if "was_home" in df.columns:
        spec["was_home"] = "mean"
    if "value" in df.columns:
        spec["value"] = "last"
    if "opponent_team" in df.columns:
        spec["opponent_team"] = "first"
    # If opponent strength was attached per-fixture, average it across a double
    # gameweek's fixtures (mirrors how build_upcoming_features handles DGWs).
    for c in ("opp_strength", "opp_strength_attack", "opp_strength_defence"):
        if c in df.columns:
            spec[c] = "mean"

    grouped = df.groupby(["player_id", "round"], as_index=False).agg(spec)
    n_fix = (
        df.groupby(["player_id", "round"]).size().rename("n_fixtures").reset_index()
    )
    out = grouped.merge(n_fix, on=["player_id", "round"], how="left")
    return out


def _attach_opponent_strength(df: pd.DataFrame, teams: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add opponent strength features (known before kickoff -> not leakage)."""
    if teams is None or "opponent_team" not in df.columns:
        for c in ["opp_strength", "opp_strength_attack", "opp_strength_defence"]:
            df[c] = np.nan
        return df
    t = teams.copy()
    t["opp_strength"] = t[["strength_overall_home", "strength_overall_away"]].mean(axis=1)
    t["opp_strength_attack"] = t[["strength_attack_home", "strength_attack_away"]].mean(axis=1)
    t["opp_strength_defence"] = t[["strength_defence_home", "strength_defence_away"]].mean(axis=1)
    cols = ["id", "opp_strength", "opp_strength_attack", "opp_strength_defence"]
    df = df.merge(t[cols], left_on="opponent_team", right_on="id", how="left")
    return df.drop(columns=["id"])


def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling / lagged form features, all shifted to exclude the target round."""
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True)
    g = df.groupby("player_id")

    for stat in _FORM_STATS:
        if stat not in df.columns:
            continue
        for w in _WINDOWS:
            df[f"{stat}_r{w}"] = g[stat].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
        # Season-to-date mean (expanding), shifted.
        df[f"{stat}_std"] = g[stat].transform(
            lambda s: s.shift(1).expanding().mean()
        )

    # Most-recent-appearance lags.
    df["last_points"] = g["total_points"].transform(lambda s: s.shift(1))
    df["last_minutes"] = g["minutes"].transform(lambda s: s.shift(1))
    # Appearances so far (0 for a player's first row).
    df["games_played"] = g.cumcount()
    return df


def build_feature_table(base_dir: str = "data") -> pd.DataFrame:
    """Build the leakage-free (player, round) feature table.

    Returns a DataFrame ready for walk-forward training/evaluation.
    """
    raw = _load_player_gw(base_dir)
    pr = _read_players_raw(base_dir)
    teams = _read_teams(base_dir)

    # Attach opponent strength per fixture, BEFORE aggregation, so double gameweeks
    # average the two opponents' strength rather than keeping only the first.
    raw = _attach_opponent_strength(raw, teams)
    pr_round = _aggregate_to_player_round(raw)

    # Attach static attributes (position, club, name) from players_raw.
    if pr is not None:
        meta_cols = ["id", "element_type", "team", "first_name", "second_name"]
        meta = pr[[c for c in meta_cols if c in pr.columns]].copy()
        pr_round = pr_round.merge(meta, left_on="player_id", right_on="id", how="left")
        pr_round = pr_round.drop(columns=["id"])
        pr_round["name"] = (
            pr_round["first_name"].fillna("") + " " + pr_round["second_name"].fillna("")
        ).str.strip()
    else:
        pr_round["element_type"] = np.nan
        pr_round["team"] = np.nan
        pr_round["name"] = "player_" + pr_round["player_id"].astype(str)

    # Drop managers and rows with no position.
    pr_round = pr_round[pr_round["element_type"].isin(PLAYER_POSITIONS)].copy()

    pr_round = _add_form_features(pr_round)

    # Targets.
    pr_round["target_points"] = pr_round["total_points"]
    pr_round["target_appeared"] = (pr_round["minutes"] >= 1).astype(int)

    logger.info(
        "Feature table: %d player-round rows, rounds %d-%d, %d players",
        len(pr_round), pr_round["round"].min(), pr_round["round"].max(),
        pr_round["player_id"].nunique(),
    )
    return pr_round


def _upcoming_fixture_context(
    fixtures: pd.DataFrame, teams: Optional[pd.DataFrame], target_round: int
) -> pd.DataFrame:
    """Per-team fixture context for an upcoming round (known before kickoff).

    Returns one row per team with: n_fixtures (0=blank, 2=double GW), was_home
    (mean), and opponent strength averaged across that round's fixture(s).
    """
    fx = fixtures[fixtures["event"] == target_round]
    strength = None
    if teams is not None:
        t = teams.copy()
        t["s_all"] = t[["strength_overall_home", "strength_overall_away"]].mean(axis=1)
        t["s_att"] = t[["strength_attack_home", "strength_attack_away"]].mean(axis=1)
        t["s_def"] = t[["strength_defence_home", "strength_defence_away"]].mean(axis=1)
        strength = t.set_index("id")[["s_all", "s_att", "s_def"]].to_dict("index")

    rows: dict[int, dict] = {}
    for _, f in fx.iterrows():
        for team, opp, is_home in ((f["team_h"], f["team_a"], 1), (f["team_a"], f["team_h"], 0)):
            r = rows.setdefault(int(team), {"home": [], "opps": []})
            r["home"].append(is_home)
            r["opps"].append(int(opp))

    out = []
    for team, r in rows.items():
        opp_all = opp_att = opp_def = np.nan
        if strength is not None:
            vals = [strength.get(o, {}) for o in r["opps"]]
            opp_all = np.nanmean([v.get("s_all", np.nan) for v in vals]) if vals else np.nan
            opp_att = np.nanmean([v.get("s_att", np.nan) for v in vals]) if vals else np.nan
            opp_def = np.nanmean([v.get("s_def", np.nan) for v in vals]) if vals else np.nan
        out.append({
            "team": team, "n_fixtures": len(r["home"]),
            "was_home": float(np.mean(r["home"])),
            "opp_strength": opp_all, "opp_strength_attack": opp_att,
            "opp_strength_defence": opp_def,
        })
    return pd.DataFrame(out)


def _availability_multiplier(feat: pd.DataFrame) -> np.ndarray:
    """Injury/suspension downweight for a LIVE forecast, in [0, 1].

    Uses the FPL API's current `status` (a=available, d=doubtful, i=injured,
    s=suspended, u/n=unavailable) and `chance_of_playing_next_round` (0-100, null
    when fully fit or fully out). These are known before kickoff, so applying them
    is not leakage. Only meaningful on live data; on a historical snapshot they
    reflect end-of-snapshot availability, so treat the offline demo as illustrative.
    """
    n = len(feat)
    chance = feat["chance_of_playing_next_round"] if "chance_of_playing_next_round" in feat else pd.Series([np.nan] * n)
    status = feat["status"] if "status" in feat else pd.Series(["a"] * n)
    out = np.ones(n)
    status = status.fillna("a").to_numpy()
    chance = chance.to_numpy(dtype=float)
    for i in range(n):
        if not np.isnan(chance[i]):
            out[i] = chance[i] / 100.0
        elif status[i] in ("i", "s", "u", "n"):
            out[i] = 0.0
        elif status[i] == "d":
            out[i] = 0.5
    return out


def build_prior_profiles(
    prior_base_dir: str, current_base_dir: Optional[str] = None, n_price_buckets: int = 5
) -> pd.DataFrame:
    """Per-player PRIOR-SEASON form profile, used to seed a new season's cold start.

    For each form stat we compute the player's per-game average over the previous
    season (``{stat}_pg``). Players with no prior-season FPL history -- new
    signings, promoted-club and youth players -- get a fallback profile: the
    average per-game stats of players in the same position and price bucket. The
    returned table covers every CURRENT player, so it can always be joined.

    In production ``prior_base_dir`` points at last season's ingested data (or use
    each player's ``history_past`` from the FPL API). Offline, the current season
    can be passed as a stand-in to exercise the mechanism.
    """
    if current_base_dir is None:
        current_base_dir = prior_base_dir

    prior = _aggregate_to_player_round(_load_player_gw(prior_base_dir))
    stats = [s for s in _FORM_STATS if s in prior.columns]

    # Per-player per-game means over the prior season (includes benched rounds).
    per_player = prior.groupby("player_id")[stats].mean()
    per_player.columns = [f"{c}_pg" for c in per_player.columns]
    per_player["n_games_prior"] = prior.groupby("player_id").size()

    # Current roster (who needs a profile) with position and price.
    cur = _read_players_raw(current_base_dir)
    if cur is None:
        raise FileNotFoundError("players_raw.csv required to build prior profiles.")
    price_col = "now_cost" if "now_cost" in cur.columns else None
    roster = cur[["id", "element_type"]].copy()
    roster["price"] = cur[price_col] if price_col else 0.0
    roster = roster[roster["element_type"].isin(PLAYER_POSITIONS)]
    roster = roster.merge(per_player, left_on="id", right_index=True, how="left")

    # Position x price-bucket fallback means for players missing a prior profile.
    pg_cols = [f"{s}_pg" for s in stats]
    roster["bucket"] = (
        roster.groupby("element_type")["price"]
        .transform(lambda s: pd.qcut(s.rank(method="first"),
                                     max(1, min(n_price_buckets, len(s))), labels=False))
    )
    fallback = roster.groupby(["element_type", "bucket"])[pg_cols].mean()
    have_prior = roster["n_games_prior"].notna()
    for col in pg_cols:
        fill = roster.set_index(["element_type", "bucket"]).index.map(fallback[col])
        roster[col] = np.where(roster[col].notna(), roster[col], np.asarray(fill, dtype=float))
    # Last-resort: global mean for any still-missing cell.
    for col in pg_cols:
        roster[col] = roster[col].fillna(roster[col].mean())

    roster["n_games_prior"] = roster["n_games_prior"].fillna(0)
    roster["has_fpl_prior"] = have_prior.values
    out = roster.rename(columns={"id": "player_id"})
    logger.info("Prior profiles: %d players (%d with real FPL history, %d via fallback).",
                len(out), int(out["has_fpl_prior"].sum()), int((~out["has_fpl_prior"]).sum()))
    return out[["player_id", "n_games_prior", "has_fpl_prior"] + pg_cols]


def seed_cold_start(feat: pd.DataFrame, profiles: pd.DataFrame, k0: float = 3.0) -> pd.DataFrame:
    """Blend within-season form toward the prior-season profile (shrinkage).

    weight w = n / (n + k0) on the within-season value, (1-w) on the prior, where
    n = games played this season. At GW1 (n=0) features are 100% prior; the prior
    fades out as the new season accumulates games. NaN within-season values (no
    data yet) fall back entirely to the prior.
    """
    feat = feat.merge(profiles, on="player_id", how="left")
    n = feat["games_played"].fillna(0).to_numpy(dtype=float)
    w = n / (n + k0)  # within-season weight in [0,1)

    for stat in _FORM_STATS:
        prior_col = f"{stat}_pg"
        if prior_col not in feat.columns:
            continue
        prior = feat[prior_col].to_numpy(dtype=float)
        for col in (f"{stat}_r3", f"{stat}_r5", f"{stat}_std"):
            if col not in feat.columns:
                continue
            within = feat[col].to_numpy(dtype=float)
            have_within = ~np.isnan(within)
            blended = np.where(have_within, w * within, 0.0) + np.where(
                have_within, (1.0 - w) * prior, prior
            )
            # Only override where we actually have a prior to seed with.
            feat[col] = np.where(np.isnan(prior), within, blended)

    # Seed the lag features when this season has none yet.
    if "total_points_pg" in feat.columns:
        feat["last_points"] = feat["last_points"].fillna(feat["total_points_pg"])
    if "minutes_pg" in feat.columns:
        feat["last_minutes"] = feat["last_minutes"].fillna(feat["minutes_pg"])
    return feat.drop(columns=[c for c in feat.columns if c.endswith("_pg")]
                     + ["n_games_prior", "has_fpl_prior"], errors="ignore")


def build_upcoming_features(
    base_dir: str = "data",
    target_round: Optional[int] = None,
    prior_profiles: Optional[pd.DataFrame] = None,
    cold_start_k0: float = 3.0,
) -> pd.DataFrame:
    """Build feature rows for a NOT-YET-PLAYED gameweek (a genuine forecast).

    For each player, form features are computed as their state *as of now* (over
    rounds strictly before ``target_round``), and the upcoming fixture's context
    (opponent, home/away, difficulty) is attached -- all of which is legitimately
    known in advance. Returns one row per player who has a fixture in the round.

    If ``target_round`` is None it defaults to the next round after the last one
    with played data (max played round + 1).
    """
    raw = _load_player_gw(base_dir)
    pr = _read_players_raw(base_dir)
    teams = _read_teams(base_dir)
    fixtures = pd.read_csv(os.path.join(base_dir, "fixtures.csv"))

    pr_round = _aggregate_to_player_round(raw)
    if target_round is None:
        target_round = int(pr_round["round"].max()) + 1
    hist = pr_round[pr_round["round"] < target_round].sort_values(["player_id", "round"])

    # Per-player "as of now" form (mirrors the shifted training-time features).
    recs = []
    for pid, grp in hist.groupby("player_id"):
        rec = {"player_id": pid, "games_played": len(grp)}
        for stat in _FORM_STATS:
            if stat not in grp.columns:
                continue
            s = grp[stat]
            rec[f"{stat}_r3"] = s.tail(3).mean()
            rec[f"{stat}_r5"] = s.tail(5).mean()
            rec[f"{stat}_std"] = s.mean()
        rec["last_points"] = grp["total_points"].iloc[-1]
        rec["last_minutes"] = grp["minutes"].iloc[-1]
        recs.append(rec)
    feat = pd.DataFrame(recs)

    # Static attributes (position, club, price, name) from players_raw.
    if pr is None:
        raise FileNotFoundError("players_raw.csv required to build upcoming features.")
    meta_cols = ["id", "element_type", "team", "now_cost", "first_name", "second_name"]
    for c in ("status", "chance_of_playing_next_round"):  # availability signals (live)
        if c in pr.columns:
            meta_cols.append(c)
    meta = pr[meta_cols].copy()
    meta["name"] = (meta["first_name"].fillna("") + " " + meta["second_name"].fillna("")).str.strip()
    # Current price (*10), known before kickoff. NB: for a live next-GW forecast this is
    # correct; when back-forecasting a mid-season round it uses the latest price, not the
    # price as it was that round (a minor mismatch that only affects the offline demo).
    meta["value"] = meta["now_cost"]
    if feat.empty:  # no within-season history at all (e.g. forecasting GW1)
        feat = pd.DataFrame({"player_id": meta["id"], "games_played": 0})
    feat = feat.merge(meta, left_on="player_id", right_on="id", how="right").drop(columns=["id"])
    feat = feat[feat["element_type"].isin(PLAYER_POSITIONS)].copy()
    feat["games_played"] = feat["games_played"].fillna(0)

    # Ensure every form/lag column exists even with little/no history, so the
    # cold-start seeding (below) and the model see a consistent schema.
    for stat in _FORM_STATS:
        for col in (f"{stat}_r3", f"{stat}_r5", f"{stat}_std"):
            if col not in feat.columns:
                feat[col] = np.nan
    for col in ("last_points", "last_minutes"):
        if col not in feat.columns:
            feat[col] = np.nan

    # Attach the upcoming fixture context; keep only players with a fixture.
    ctx = _upcoming_fixture_context(fixtures, teams, target_round)
    feat = feat.merge(ctx, on="team", how="inner")
    feat["round"] = target_round
    feat["availability"] = _availability_multiplier(feat)

    # Cold start: early in a season the within-season form is thin/empty, so seed
    # it from the previous-season profile (fades out as games accumulate).
    if prior_profiles is not None:
        feat = seed_cold_start(feat, prior_profiles, k0=cold_start_k0)

    logger.info("Upcoming round %d: %d players with a fixture (blank/double GWs handled%s).",
                target_round, len(feat),
                ", cold-start seeded" if prior_profiles is not None else "")
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model input columns (everything known before the round kicks off)."""
    cols: list[str] = []
    for stat in _FORM_STATS:
        for w in _WINDOWS:
            c = f"{stat}_r{w}"
            if c in df.columns:
                cols.append(c)
        c = f"{stat}_std"
        if c in df.columns:
            cols.append(c)
    cols += ["last_points", "last_minutes", "games_played"]
    # Pre-match context (legitimately known in advance).
    cols += ["was_home", "n_fixtures", "value", "element_type",
             "opp_strength", "opp_strength_attack", "opp_strength_defence"]
    return [c for c in cols if c in df.columns]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = build_feature_table()
    feats = feature_columns(table)
    print(f"\nRows: {len(table)} | Features: {len(feats)}")
    print("Feature columns:", feats)
    print(table[["player_id", "name", "round", "element_type",
                 "target_points", "target_appeared", "last_points",
                 "total_points_r5"]].head(12).to_string())
