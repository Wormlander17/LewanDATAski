"""Make our submission.

    python -m possession.run                    # writes evaluation/results/<team>.csv
    python -m possession.run --name myteam

Steps
  1. Load tracking and Smart Tagging.
  2. Build evidence: ball control, team shape, tags, ball out / still, kickoffs.
  3. Train the two self supervised models (team, dead ball), out of fold.
  4. Combine the evidence into H / A / D scores per frame.
  5. Viterbi decode each half into clean possession chunks.
  6. Write the CSV. Ungraded frames get X.

All settings are in PARAMS below. They were chosen from football common sense and NOT tuned on
the public 5 minutes of ground truth (that is against the rules, and it picks the wrong answer).
"""

import argparse
import os
import sys

import numpy as np

from .load import load_tracking, load_tags, ROOT
from . import features as F
from .models import (out_of_fold, team_pseudo_labels, dead_pseudo_labels, balance)
from .decode import viterbi_min_duration, emissions

PARAMS = {
    # how much each piece of evidence counts (log odds units)
    "shape_clip": 3.0,       # team model, from player shape only
    "control": 3.0,          # last team in control (loose ball stays with it), fades after
    "control_fade": 2.0,     # seconds for that to fade so shape and tags take over
    "tag": 1.0,              # Smart Tagging team phase
    # restart taker, weight = log odds of how often it was right in our own check (no answer key)
    "taker_visible": 2.9,    # ~95% right
    "taker_hidden": 1.4,     # ~80% right
    "dead_clip": 4.0,        # dead ball model
    "stoppage": 8.0,         # inside a stoppage we can see (ball out / placed), or just after
    "ball_missing": 0.5,     # tracker lost the ball (mild hint of a stoppage)
    # decoder: evidence is counted per second, a switch costs this many units
    "scale": 1.0 / F.FPS,
    "switch_team": 1.0,
    "switch_dead": 3.0,
    # shortest allowed run, in seconds: a possession needs real control; even the quickest
    # restart (a quick throw in) takes about 4 s
    "min_possession": 1.0,
    "min_dead": 4.0,
}


def logit(p, clip):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


EVIDENCE_CACHE = os.path.join(ROOT, "data", "cache_evidence.npz")


def build_evidence(verbose=True, use_cache=True):
    if use_cache and os.path.exists(EVIDENCE_CACHE):
        z = np.load(EVIDENCE_CACHE)
        ev = {k: z[k] for k in z.files}
        ev["n"] = int(ev["n"])
        return ev
    ball, players, teams = load_tracking()
    tags = load_tags()
    n = len(ball)
    ball, players = F.rotate(ball, players, teams)

    ctrl, _, _ = F.ball_control(ball, players, teams)
    out, still, seen, ball_speed = F.ball_state(ball)
    stopped, restart_frames = F.stoppages(ball, out, still, seen, ball_speed)
    restart = np.zeros(n, dtype=bool)
    for r in restart_frames:
        restart[r:r + F.FPS] = True          # the second after a restart is live play
    tag_vote = F.tag_evidence(tags, n)

    # Before the kickoff of each half the ball is dead. Found from the 'Kick Off (Start)' tags.
    kickoffs = F.kickoff_frames(tags, ball)
    before_kickoff = np.zeros(n, dtype=bool)
    for kind, frame in kickoffs:
        if kind == "kick off (start)":
            half_start = max(a for a, _ in F.HALVES if a <= frame)
            before_kickoff[half_start:frame] = True

    taker_visible, taker_hidden = F.restart_takers(
        ball, players, teams, sorted(set(restart_frames) | {f for _, f in kickoffs}))
    for frame, side in F.conceding_kickoff(tags, kickoffs).items():
        taker_hidden[frame:frame + 2 * F.FPS] += side

    X, names = F.team_shape(players, teams)

    team_y = team_pseudo_labels(ctrl, out)
    p_home = out_of_fold(X, team_y, balance(team_y))

    dead_y = dead_pseudo_labels(team_y, stopped, before_kickoff)
    # Dead ball model sees tempo only, so it cannot confuse "where teams stand" with "is play on".
    tempo = X[:, F.tempo_columns(names)]
    p_dead = out_of_fold(tempo, dead_y, balance(dead_y))
    # Where the ball is visible nearby in time, a second model also sees how the ball is played
    # (passes between different players = live, even when everyone is walking).
    Xb, ball_near = F.ball_play(ball, players, out, still)
    yb = np.where(ball_near, dead_y, -1).astype(np.int8)
    p_dead_ball = out_of_fold(np.column_stack([tempo, Xb]), yb, balance(yb))
    p_dead = np.where(ball_near & ~np.isnan(p_dead_ball), p_dead_ball, p_dead)

    if verbose:
        g = F.half_of(n) > 0
        m = g & (team_y >= 0)
        acc = np.mean((p_home[m] > 0.5) == (team_y[m] == 1))
        m2 = g & (dead_y >= 0)
        acc2 = np.mean((p_dead[m2] > 0.5) == (dead_y[m2] == 1))
        print("pseudo labels: team {} frames, dead {} / live {} frames".format(
            int(m.sum()), int((dead_y[g] == 1).sum()), int((dead_y[g] == 0).sum())))
        print("out of fold agreement with pseudo labels: team {:.3f}, dead {:.3f}".format(
            acc, acc2))

    ev = {
        "n": n, "ctrl": ctrl, "out": out, "still": still, "seen": seen,
        "tag_vote": tag_vote, "before_kickoff": before_kickoff,
        "stopped": stopped, "restart": restart,
        "taker_visible": taker_visible, "taker_hidden": taker_hidden,
        "p_home": p_home, "p_dead": p_dead,
    }
    np.savez_compressed(EVIDENCE_CACHE, **ev)
    return ev


def decide(ev, P=PARAMS):
    n = ev["n"]
    # Direct ball evidence (who controlled it last) leads; the slow phase evidence (team shape,
    # tags) only fills in once that has faded, e.g. deep inside a ball gap.
    side, fresh = F.last_controller(ev["ctrl"], ev["stopped"], fade=P["control_fade"] * F.FPS)
    phase = logit(ev["p_home"], P["shape_clip"]) + P["tag"] * ev["tag_vote"]
    team = P["control"] * side * fresh + (1.0 - fresh) * phase
    team += P["taker_visible"] * ev["taker_visible"] + P["taker_hidden"] * ev["taker_hidden"]

    dead = logit(ev["p_dead"], P["dead_clip"])
    dead += P["ball_missing"] * (~ev["seen"])
    dead[ev["stopped"]] = P["stoppage"]
    dead[ev["restart"] & ~ev["stopped"]] = -P["stoppage"]
    dead[ev["before_kickoff"]] = 20.0     # certain

    team = np.nan_to_num(team)
    dead = np.nan_to_num(dead)
    labels = np.full(n, "X", dtype="<U1")
    for a, b in F.HALVES:
        e = emissions(team[a:b], dead[a:b], P["scale"])
        min_len = [P["min_possession"] * F.FPS] * 2 + [P["min_dead"] * F.FPS]
        path = viterbi_min_duration(e, P["switch_team"], P["switch_dead"], min_len)
        labels[a:b] = np.array(["H", "A", "D"])[path]
    return labels


def describe(labels):
    """Possession counts and durations, the numbers a coach would sanity check."""
    g = F.half_of(len(labels)) > 0
    runs = []
    for a, b in F.HALVES:
        seg = labels[a:b]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(seg)]
        runs += [(seg[s], (e - s) / F.FPS) for s, e in zip(starts, ends)]
    poss = [d for lab, d in runs if lab in "HA"]
    dead = [d for lab, d in runs if lab == "D"]
    share = {k: np.mean(labels[g] == k) for k in "HAD"}
    print("possessions {}  mean {:.1f} s  median {:.1f} s".format(
        len(poss), np.mean(poss), np.median(poss)))
    print("dead spells {}  mean {:.1f} s".format(len(dead), np.mean(dead) if dead else 0))
    print("share  H {H:.3f}  A {A:.3f}  D {D:.3f}".format(**share))


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="barcathon_gk")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    ap.add_argument("--fresh", action="store_true", help="rebuild evidence, ignore the cache")
    args = ap.parse_args()

    ev = build_evidence(use_cache=not args.fresh)
    labels = decide(ev)
    describe(labels)
    path = os.path.join(args.out, args.name + ".csv")
    write(path, labels)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
