"""
Live data ingestion from the OFFICIAL Fantasy Premier League API.

Why this replaces the old ingestion
------------------------------------
The previous version scraped a hardcoded `2024-25/` folder from the community
repo vaastav/Fantasy-Premier-League. That repo STOPPED its weekly updates after
the 2024-25 season, so it cannot feed a model for the upcoming season. The
official FPL API is free, live, and authoritative, so we make it the primary
feed. Raw JSON is cached to disk so re-runs are reproducible and offline-friendly.

Endpoints used (no API key required):
  * bootstrap-static/        -> players (elements), teams, positions, prices
  * fixtures/                -> all fixtures + difficulty
  * element-summary/{id}/    -> per-player per-gameweek history (incl. `round`,
                                minutes, total_points, expected_goals, ...)

Output layout (identical to what data_processing.py expects):
  data/players_raw.csv          (elements / static player attributes)
  data/teams.csv                (team strengths)
  data/player_idlist.csv        (id, first_name, second_name)
  data/fixtures.csv
  data/players/<First_Second_id>/gw.csv   (one file per player, with `round`)

Note: requires network access to fantasy.premierleague.com.
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-prediction-pipeline)"}
REQUEST_PAUSE = 0.3          # be polite to the API
TIMEOUT = 15


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get_json(session: requests.Session, url: str, cache_path: Optional[str] = None,
              use_cache: bool = True) -> Optional[dict]:
    """GET JSON with on-disk caching and basic retry."""
    if cache_path and use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
            return data
        except Exception as exc:
            logger.warning("GET %s failed (attempt %d/3): %s", url, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    return None


def ingest_data(base_dir: str = "data", use_cache: bool = True,
                max_players: Optional[int] = None) -> None:
    """Download current-season FPL data and write the on-disk layout."""
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = os.path.join(base_dir, "cache")
    session = _session()

    # --- bootstrap-static: players, teams ---
    boot = _get_json(session, f"{BASE}/bootstrap-static/",
                     os.path.join(cache_dir, "bootstrap-static.json"), use_cache)
    if boot is None:
        raise RuntimeError("Could not reach the FPL API (bootstrap-static). Check network.")

    elements = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    elements.to_csv(os.path.join(base_dir, "players_raw.csv"), index=False)
    teams.to_csv(os.path.join(base_dir, "teams.csv"), index=False)
    elements[["id", "first_name", "second_name"]].to_csv(
        os.path.join(base_dir, "player_idlist.csv"), index=False)
    logger.info("Saved %d players and %d teams", len(elements), len(teams))

    # --- fixtures ---
    fixtures = _get_json(session, f"{BASE}/fixtures/",
                         os.path.join(cache_dir, "fixtures.json"), use_cache)
    if fixtures is not None:
        pd.DataFrame(fixtures).to_csv(os.path.join(base_dir, "fixtures.csv"), index=False)

    # --- per-player gameweek history ---
    players_dir = os.path.join(base_dir, "players")
    os.makedirs(players_dir, exist_ok=True)
    ids = elements["id"].tolist()
    if max_players:
        ids = ids[:max_players]

    failed = []
    for n, pid in enumerate(ids, 1):
        row = elements.loc[elements["id"] == pid].iloc[0]
        folder = f"{row['first_name']}_{row['second_name']}_{int(pid)}".replace("/", "_")
        summary = _get_json(session, f"{BASE}/element-summary/{pid}/",
                            os.path.join(cache_dir, f"element-{pid}.json"), use_cache)
        if summary is None or not summary.get("history"):
            failed.append(pid)
            continue
        hist = pd.DataFrame(summary["history"])
        hist["player_id"] = pid
        # FPL history uses `round` already; keep it as the time axis.
        out_dir = os.path.join(players_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        hist.to_csv(os.path.join(out_dir, "gw.csv"), index=False)
        if not use_cache:
            time.sleep(REQUEST_PAUSE)
        if n % 100 == 0:
            logger.info("  ...ingested %d/%d players", n, len(ids))

    logger.info("Ingestion complete. %d players written, %d failed.",
                len(ids) - len(failed), len(failed))
    if failed:
        logger.warning("Failed player ids (no history / fetch error): %s", failed[:20])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_data()
