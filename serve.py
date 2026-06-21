"""
Zero-dependency web server for the FPL prediction + decision app.

Uses only the Python standard library (http.server) so it runs anywhere with no
extra installs. On startup it runs the pipeline once (train -> forecast the next
gameweek -> default optimal squad) and caches the result; the browser UI then:
  * GET /api/data                 -> meta, players, teams, default squad
  * GET /api/optimize?budget=&lock=&exclude=  -> re-run the squad ILP live

Run:  python serve.py            (then open http://localhost:8000)
Env:  FPL_DATA_DIR, FPL_PRIOR_SEASON_DIR (same meaning as main.py), PORT.
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd

from data_processing import (
    build_feature_table, build_upcoming_features, build_prior_profiles,
    feature_columns, next_unfinished_round, POSITION_MAP,
)
from model import FPLPointsModel, make_predictor, engine_name
from validation import walk_forward_predict, compute_metrics, BASELINES
from optimizer import optimize_squad

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("FPL_DATA_DIR", "data")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MIN_TRAIN_ROUNDS = 8

STATE: dict = {}  # cached pipeline output


def _quick_metric(table, feats) -> dict | None:
    """Headline accuracy on the last 3 rounds: model vs predict-last baseline."""
    rounds = sorted(table["round"].unique())
    if len(rounds) < MIN_TRAIN_ROUNDS + 2:
        return None
    test = rounds[-3:]
    model = compute_metrics(walk_forward_predict(table, make_predictor(feats), test, MIN_TRAIN_ROUNDS))
    base = compute_metrics(walk_forward_predict(table, BASELINES["predict_last"], test, MIN_TRAIN_ROUNDS))
    return {"mae": round(model["MAE"], 3), "rmse": round(model["RMSE"], 3),
            "spearman": round(model["spearman"], 3),
            "baselineMae": round(base["MAE"], 3), "baselineName": "last GW",
            "rounds": [int(r) for r in test]}


def precompute() -> None:
    logger.info("Running pipeline (this takes ~30-60s)...")
    teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
    team_name = dict(zip(teams["id"], teams["name"]))
    team_short = dict(zip(teams["id"], teams["short_name"]))

    table = build_feature_table(DATA_DIR)
    feats = feature_columns(table)
    played = sorted(table["round"].unique())
    gw = next_unfinished_round(DATA_DIR)
    prior_dir = os.environ.get("FPL_PRIOR_SEASON_DIR")
    opener = len(played) < MIN_TRAIN_ROUNDS + 2

    if not opener:
        metric = _quick_metric(table, feats)
        model = FPLPointsModel().fit(table, feats)
        profiles = build_prior_profiles(prior_dir, DATA_DIR) if prior_dir else None
        mode = "midseason"
    else:
        metric = None
        if not prior_dir:
            raise SystemExit("Season opener needs FPL_PRIOR_SEASON_DIR=<last-season-dir>.")
        prior_table = build_feature_table(prior_dir)
        model = FPLPointsModel().fit(prior_table, feature_columns(prior_table))
        profiles = build_prior_profiles(prior_dir, DATA_DIR)
        mode = "opener"

    up = build_upcoming_features(DATA_DIR, target_round=gw, prior_profiles=profiles)
    up["raw_pred"] = model.predict(up)
    up["pred"] = (up["raw_pred"] * up["availability"]).round(3)

    players = []
    for _, r in up.iterrows():
        price = round(r["value"] / 10.0, 1)
        players.append({
            "id": int(r["player_id"]),
            "name": r["name"],
            "pos": int(r["element_type"]),
            "posLabel": POSITION_MAP.get(int(r["element_type"]), "?"),
            "team": int(r["team"]) if pd.notna(r["team"]) else 0,
            "teamShort": team_short.get(r["team"], "?"),
            "teamName": team_name.get(r["team"], "?"),
            "price": price,
            "pred": round(float(r["pred"]), 2),
            "avail": round(float(r["availability"]), 2),
            "form": round(float(r["total_points_r5"]), 1) if pd.notna(r["total_points_r5"]) else None,
            "value": round(float(r["pred"]) / price, 2) if price else 0.0,
            "home": bool(r["was_home"] >= 0.5) if pd.notna(r["was_home"]) else None,
        })

    pdf = up.rename(columns={"element_type": "pos"}).copy()
    pdf["price"] = pdf["value"] / 10.0
    STATE.update({
        "players_df": pdf[["player_id", "name", "pos", "team", "price", "pred"]],
        "meta": {
            "season": os.environ.get("FPL_SEASON_LABEL", "2024-25 archive (demo)"),
            "gw": gw, "mode": mode, "engine": engine_name(),
            "nPlayers": len(players), "metric": metric, "budget": 100.0,
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "players": players,
        "teams": {int(k): v for k, v in team_name.items()},
    })
    STATE["squad"] = _solve()
    logger.info("Ready: GW%s, %d players, mode=%s.", gw, len(players), mode)


def _solve(budget: float = 100.0, lock: list | None = None, exclude: list | None = None) -> dict:
    sol = optimize_squad(STATE["players_df"], budget=budget, force_in=lock, exclude=exclude)
    sq = sol.squad
    return {
        "cost": round(sol.total_cost, 1),
        "xiPoints": round(sol.xi_expected_points, 1),
        "budget": budget,
        "picks": [{"id": int(r.player_id), "inXi": bool(r.in_xi), "isCaptain": bool(r.is_captain)}
                  for r in sq.itertuples()],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter logs
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/api/data",):
            payload = {k: STATE[k] for k in ("meta", "players", "teams", "squad")}
            return self._send(200, json.dumps(payload))
        if u.path == "/api/optimize":
            q = parse_qs(u.query)
            try:
                budget = float(q.get("budget", ["100"])[0])
                lock = [int(x) for x in q.get("lock", [""])[0].split(",") if x]
                exclude = [int(x) for x in q.get("exclude", [""])[0].split(",") if x]
                return self._send(200, json.dumps(_solve(budget, lock, exclude)))
            except Exception as exc:  # infeasible / bad input
                return self._send(200, json.dumps({"error": str(exc)}))
        # static files
        path = "/index.html" if u.path == "/" else u.path
        fp = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
        if not fp.startswith(WEB_DIR) or not os.path.isfile(fp):
            return self._send(404, "not found", "text/plain")
        ext = os.path.splitext(fp)[1]
        ctype = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
                 ".json": "application/json", ".svg": "image/svg+xml"}.get(ext, "text/plain")
        with open(fp, "rb") as fh:
            self._send(200, fh.read(), ctype)


def main():
    precompute()
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    logger.info("Serving on http://localhost:%d  (Ctrl-C to stop)", port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
