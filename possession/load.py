"""Load the tactical tracking (ATD) and Smart Tagging into numpy arrays.

The result is cached in data/cache_tactical.npz so later runs take a second.
Arrays are indexed by VIDEO frame (0 ... 199318), the same index as a submission row.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1


def track_teams(meta_path):
    """Team (0 home, 1 away) of every track slot, from the ATD metadata."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def load_tracking(game=GAME):
    """ball (N,2) and players (N,S,2) in metres, NaN where missing, plus team of each slot."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    cache = os.path.join(ROOT, "data", "cache_tactical.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["ball"], z["players"], z["teams"]

    teams = track_teams(meta_path)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    slots = len(teams)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, slots, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # ATD frame_id counts from 1
            b = parts[2].split(",")
            ball[f] = (float(b[0]) * length, float(b[1]) * width)
            for s, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                g = group.split(",")
                players[f, s] = (float(g[0]) * length, float(g[1]) * width)

    np.savez_compressed(cache, ball=ball, players=players, teams=teams)
    return ball, players, teams


def load_tags(game=GAME):
    """Smart Tagging as a list of dicts: code, start, end (video frames), team, type."""
    path = require(["smart_tagging"], game)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {}
        for lab in inst.iter("label"):
            group = (lab.findtext("group") or "").strip().lower()
            tags[group] = (lab.findtext("text") or "").strip()
        team = tags.get("team", "").lower()
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": HOME if team == "barcelona" else AWAY if team.startswith("at") else None,
            "type": tags.get("type", "").lower(),
        })
    return out
