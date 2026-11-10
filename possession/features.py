"""Turn raw tracking into per-frame evidence.

Everything here is computed by code from the tracking and Smart Tagging. Nothing is hand labelled.

Coordinates are rotated per half so that the HOME team always attacks towards +x. That way a
feature like "home centroid x" means the same thing in both halves.
"""

import warnings

import numpy as np

from .load import HOME, AWAY

warnings.filterwarnings("ignore", category=RuntimeWarning)

FPS = 25
PITCH = (105.3, 68.0)
HALVES = ((11185, 90652), (110808, 190198))  # graded video frames, end exclusive

# Tag codes that name the team WITH the ball, and the ones that name the team WITHOUT it.
ATTACK_CODES = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
                "CREATING CHANCES", "LONG BALL", "BALL IN THE BOX", "PLAYERS IN THE BOX",
                "GOALS", "SET PIECES"}
DEFEND_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}

# A player "controls" the ball when he is this close and nobody from the other team is near.
CONTROL_RADIUS = 1.0      # metres
CONTEST_RADIUS = 2.0      # metres, nearest opponent must be further than this


def half_of(n):
    """Array: 1 or 2 for graded frames of each half, 0 for ungraded frames."""
    h = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(HALVES, 1):
        h[a:b] = i
    return h


def rotate(ball, players, teams):
    """Rotate each half so home attacks +x. Returns copies."""
    ball, players = ball.copy(), players.copy()
    L, W = PITCH
    for a, b in HALVES:
        home_x = np.nanmean(players[a:b, teams == HOME, 0])
        away_x = np.nanmean(players[a:b, teams == AWAY, 0])
        if home_x > away_x:  # home defends the right goal in this half, so flip it
            for arr in (ball[a:b], players[a:b]):
                arr[..., 0] = L - arr[..., 0]
                arr[..., 1] = W - arr[..., 1]
    return ball, players


def rolling_mean(x, half_width):
    """Centred moving average along axis 0 that ignores NaN."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    k = 2 * half_width + 1
    pad = [(half_width, half_width)] + [(0, 0)] * (x.ndim - 1)
    cs = np.cumsum(np.pad(filled, pad), axis=0)
    cn = np.cumsum(np.pad(valid.astype(np.float64), pad), axis=0)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:]), cs])
    cn = np.concatenate([np.zeros((1,) + cn.shape[1:]), cn])
    s = cs[k:] - cs[:-k]
    c = cn[k:] - cn[:-k]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def ball_control(ball, players, teams):
    """Per frame: HOME or AWAY if one player clearly has the ball, else -1. Also distances."""
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    dh = d[:, teams == HOME].min(1)
    da = d[:, teams == AWAY].min(1)
    ctrl = np.full(len(ball), -1, dtype=np.int8)
    ctrl[(dh < CONTROL_RADIUS) & (da > CONTEST_RADIUS)] = HOME
    ctrl[(da < CONTROL_RADIUS) & (dh > CONTEST_RADIUS)] = AWAY
    return ctrl, dh, da


def team_shape(players, teams):
    """Features describing how each team is spread out and moving. No ball used.

    Home attacks +x after rotate(), so "forward" for home is +x and for away is -x.
    """
    L, _ = PITCH
    vel = np.empty_like(players)
    vel[1:-1] = (players[2:] - players[:-2]) * (FPS / 2.0)
    vel[0] = vel[-1] = np.nan
    speed = np.linalg.norm(vel, axis=2)
    bad = speed > 12.0  # faster than a sprinter means the tracker swapped identities
    speed[bad] = np.nan
    vel[bad] = np.nan

    cols, names = [], []
    for team, sign, tag in ((HOME, 1.0, "h"), (AWAY, -1.0, "a")):
        p = players[:, teams == team]
        x, y = p[..., 0], p[..., 1]
        fwd = x if sign > 0 else L - x           # distance from own goal line
        v = vel[:, teams == team]
        s = speed[:, teams == team]
        stats = {
            "cx": np.nanmean(fwd, 1),
            "sx": np.nanstd(x, 1),
            "sy": np.nanstd(y, 1),
            "front": np.nanmax(fwd, 1),
            "back": np.sort(np.where(np.isnan(fwd), np.inf, fwd), 1)[:, 1],
            "speed": np.nanmean(s, 1),
            "fast": np.nanmean(s > 4.0, 1),
            "vfwd": np.nanmean(v[..., 0] * sign, 1),
            "n": np.sum(~np.isnan(x), 1).astype(float),
        }
        stats["back"][np.isinf(stats["back"])] = np.nan
        for k, val in stats.items():
            cols.append(val)
            names.append(tag + "_" + k)

    base = np.column_stack(cols)
    h = {n: i for i, n in enumerate(names)}

    # Tempo: in live play players sprint and move together; at a stoppage they walk anywhere.
    moving = speed > 1.0
    unit = vel / np.where(moving, speed, np.nan)[..., None]
    coherence = np.linalg.norm(np.nanmean(unit, 1), axis=1)      # 1 = all same direction
    top_speed = np.nanpercentile(np.where(np.isnan(speed), -1, speed), 90, axis=1)

    extra = np.column_stack([
        base[:, h["h_cx"]] - base[:, h["a_cx"]],
        base[:, h["h_sx"]] * base[:, h["h_sy"]] - base[:, h["a_sx"]] * base[:, h["a_sy"]],
        base[:, h["h_front"]] + base[:, h["a_front"]],
        np.nanmean(speed, 1),
        coherence,
        top_speed,
        np.nanmean(speed > 2.0, 1),
    ])
    names += ["d_cx", "d_area", "fronts", "all_speed", "coherence", "top_speed", "share_running"]
    raw = np.column_stack([base, extra])
    # The same features averaged over 1 s and 4 s: possession is a slow signal.
    feats = np.column_stack([raw, rolling_mean(raw, 12), rolling_mean(raw, 50)])
    names = names + [n + "_1s" for n in names] + [n + "_4s" for n in names]
    return feats.astype(np.float32), names


TEMPO_KEYS = ("speed", "fast", "all_speed", "coherence", "top_speed", "share_running")


def tempo_columns(names):
    """Indexes of the features that describe HOW players move, not WHERE they are."""
    return [i for i, nm in enumerate(names)
            if nm.split("_", 1)[-1].replace("_1s", "").replace("_4s", "") in TEMPO_KEYS
            or nm.replace("_1s", "").replace("_4s", "") in TEMPO_KEYS]


def tag_evidence(tags, n):
    """Per frame team vote from Smart Tagging in [-1, 1]: +1 home has it, -1 away has it."""
    votes = np.zeros(n)
    count = np.zeros(n)
    for t in tags:
        if t["team"] is None:
            continue
        if t["code"] in ATTACK_CODES:
            side = 1.0 if t["team"] == HOME else -1.0
        elif t["code"] in DEFEND_CODES:
            side = -1.0 if t["team"] == HOME else 1.0
        else:
            continue
        votes[t["start"]:t["end"]] += side
        count[t["start"]:t["end"]] += 1
    return np.where(count > 0, votes / np.maximum(count, 1), 0.0)


def kickoff_frames(tags, ball):
    """Estimated restart frame of each 'Kick Off' tag: when the ball first moves inside it."""
    v = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    out = []
    for t in tags:
        if not t["type"].startswith("kick off"):
            continue
        a, b = t["start"], t["end"]
        moving = np.where(v[a:b] > 3.0)[0]
        moving = moving[moving > 0]
        out.append((t["type"], a + int(moving[0]) if len(moving) else (a + b) // 2))
    return out


def stoppages(ball, out, still, seen, ball_speed, min_out=8, max_back=15 * FPS,
              kick_speed=4.0, kick_frames=3, max_hole=FPS, max_wait=60 * FPS,
              tempo=None, live_tempo=2.2, grace=10 * FPS):
    """Stoppages we can SEE: [start, end) spans where the ball is surely dead.

    * Ball out of the pitch for at least `min_out` frames. Dead from the first out frame.
    * Ball lying still for 2 s (placed for a free kick, goal kick, corner, kickoff). Dead back to
      when the ball was last moving (the foul or the ball going out), at most 15 s back.

    A stoppage lasts until the RESTART: a clear kick or throw (ball faster than `kick_speed` m/s
    for `kick_frames` frames in a row), never taken from behind a goal line. A keeper walking
    with the ball, or the ball being rolled back onto the pitch over the goal line, is not a
    restart. If the tracker loses
    the ball for more than `max_hole` frames, the visible stoppage stops there (the bridge below
    and the dead ball model take over). Safety brake: if `tempo` (players' mean speed, 2 s
    average) is given, a stoppage also ends once players move at live play pace (`live_tempo`), after a `grace`
    period at its start while players slow down.
    Returns a boolean mask and the list of restart frames.
    """
    n = len(ball)
    dead = np.zeros(n, dtype=bool)
    restarts = []
    v = np.nan_to_num(ball_speed, nan=0.0)
    slow = seen & (v < 2.0)
    fast = seen & (v > kick_speed)
    # A restart is the START of a clear kick or throw (fast for `kick_frames` frames in a row,
    # so jitter while a keeper walks with the ball does not count), never taken from behind a
    # goal line (only throw ins come from outside, over a touchline).
    L, _ = PITCH
    kicked = np.zeros(n, dtype=bool)
    edges = np.diff(np.r_[0, fast.astype(np.int8), 0])
    for r, r_end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if r_end - r < kick_frames or r < 10:
            continue
        x = ball[r, 0]
        behind_goal_line = x < -0.5 or x > L + 0.5
        if not behind_goal_line:
            kicked[r] = True

    def runs(mask):
        d = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))

    busy = np.zeros(n, dtype=bool) if tempo is None else np.nan_to_num(tempo) > live_tempo

    def until_restart(e, origin):
        """From frame e, walk forward to the restart. Returns (end, restarted)."""
        start, hole = e, 0
        while e < n and e - start < max_wait:
            if kicked[e] and not out[e]:
                return e, True
            if busy[e] and e - origin > grace:      # players need a few seconds to slow down
                return e, False
            hole = hole + 1 if not seen[e] else 0
            if hole > max_hole:
                return e - hole + 1, False
            e += 1
        return e, False

    for a, b in runs(out):
        if b - a < min_out:
            continue
        e, ok = until_restart(b, a)
        dead[a:e] = True
        if ok:
            restarts.append(e)

    for a, b in runs(still):
        s = a
        while s > max(0, a - max_back) and slow[s - 1] and not out[s - 1]:
            s -= 1
        e, ok = until_restart(b, s)
        dead[s:e] = True
        if ok:
            restarts.append(e)

    restarts = [r for r in restarts if not dead[r]]   # a restart inside another stoppage is not one

    # Bridge: dead just before the tracker lost the ball, and when it finds the ball again it is
    # lying still or being placed -> nothing restarted in between, so the gap is dead too.
    for a, b in runs(~seen):
        if a < FPS or b >= n - 5 or not dead[a - FPS:a].any():   # dead within 1 s before
            continue
        if dead[b] or slow[b:b + 5].all():
            e = b
            while e < n and slow[e]:
                e += 1
            last = a - FPS + int(np.flatnonzero(dead[a - FPS:a])[-1])  # last dead frame
            dead[last:e] = True
            restarts = [r for r in restarts if not last <= r < e]
            if e < n and seen[e]:
                restarts.append(e)

    return dead, sorted(set(restarts))


def ball_state(ball):
    """Ball out of pitch, ball still (stationary for 2 s), ball seen, ball speed."""
    L, W = PITCH
    seen = ~np.isnan(ball[:, 0])
    margin = 0.5
    out = seen & ((ball[:, 0] < -margin) | (ball[:, 0] > L + margin) |
                  (ball[:, 1] < -margin) | (ball[:, 1] > W + margin))
    v = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    slow = (v < 0.5).astype(float)
    slow[np.isnan(v)] = np.nan
    still = rolling_mean(slow, 25) > 0.95
    still &= seen
    return out, still, seen, v


def player_speed(players):
    """Speed of every track slot in m/s, NaN where unknown or impossible (identity swap)."""
    vel = np.empty_like(players)
    vel[1:-1] = (players[2:] - players[:-2]) * (FPS / 2.0)
    vel[0] = vel[-1] = np.nan
    speed = np.linalg.norm(vel, axis=2)
    speed[speed > 12.0] = np.nan
    return speed


def restart_takers(ball, players, teams, restarts, after=2 * FPS):
    """Who takes each restart, as a per frame vote: +1 home, -1 away, 0 nothing.

    visible : the restart is seen, so the taker is the player on the ball (within 2.5 m) just
              before it moves. Checked against our own control labels: right about 95% of the time.
    hidden  : the ball is often lost during a stoppage, but the taker still shows up in the
              players: during a slow moment one team has a player standing just off the pitch
              (throw in) or at a corner flag. Checked the same way: right about 80% of the time.
    The vote covers the taker moment and the next `after` frames, when his team has the ball.
    """
    L, W = PITCH
    n = len(ball)
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    dh, da = d[:, teams == HOME].min(1), d[:, teams == AWAY].min(1)
    near_home = dh < da
    nearest = np.minimum(dh, da)

    visible = np.zeros(n)
    for r in restarts:
        w = slice(max(0, r - 10), min(n, r + 3))
        ok = nearest[w] < 2.5
        if ok.any():
            side = 1.0 if near_home[w][ok].mean() >= 0.5 else -1.0
            visible[r:r + after] = side

    speed = player_speed(players)
    tempo = rolling_mean(np.nanmean(speed, 1), FPS)
    x, y = players[..., 0], players[..., 1]
    margin = 0.3
    off = (y < -margin) | (y > W + margin) | (x < -margin) | (x > L + margin)
    off |= (np.minimum(x, L - x) < 3) & (np.minimum(y, W - y) < 3)      # at a corner flag
    off &= speed < 1.5                                                  # standing, not running
    off_home = off[:, teams == HOME].any(1)
    off_away = off[:, teams == AWAY].any(1)
    one = (off_home ^ off_away) & (np.nan_to_num(tempo, nan=9.0) < 1.3)
    hidden = np.zeros(n)
    edges = np.diff(np.r_[0, one.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < FPS:
            continue
        side = 1.0 if off_home[a:b].mean() >= 0.5 else -1.0
        hidden[a:b + after] = side
    return visible, hidden


def conceding_kickoff(tags, kickoffs, after=2 * FPS):
    """After a goal the team that conceded kicks off. GOALS tags name the scoring team.

    Vote +1 home / -1 away from each 'kick off (after goal)' restart for `after` frames.
    """
    n_vote = {}
    goals = [t for t in tags if t["code"] == "GOALS" and t["team"] is not None]
    for kind, frame in kickoffs:
        if kind != "kick off (after goal)":
            continue
        before = [g for g in goals if g["start"] <= frame <= g["end"] + 10 * FPS]
        if before:
            scorer = before[-1]["team"]
            n_vote[frame] = 1.0 if scorer == AWAY else -1.0     # the other team kicks off
    return n_vote


def last_controller(ctrl, stopped, min_run=3, max_hole=3, min_keep=12, fade=2 * FPS):
    """The rule from the challenge: a loose ball stays with the last team that controlled it.

    ctrl    : per frame HOME / AWAY / -1 from ball_control().
    A touch is a run of at least `min_run` control frames (holes of up to `max_hole` allowed).
    A team WINS the ball when its consecutive touches (before the other team touches it) add up
    to `min_keep` frames (half a second), or come in two or more touches (it played on, e.g.
    intercepted and passed). A single short touch, like a header won but not kept, is ignored
    and the ball stays with the team that had it. Ownership starts at the first touch.

    Returns
      side      : +1 home, -1 away, 0 nobody yet (or reset by a stoppage)
      freshness : 1 while the team is touching the ball, fading to 0 over `fade` frames after
    """
    n = len(ctrl)
    side = np.zeros(n)
    fresh = np.zeros(n)
    idx = np.flatnonzero(ctrl >= 0)
    runs = []
    if len(idx):
        start = prev = idx[0]
        for i in idx[1:]:
            if ctrl[i] != ctrl[prev] or i - prev > max_hole + 1:
                runs.append((start, prev + 1, int(ctrl[prev])))
                start = i
            prev = i
        runs.append((start, prev + 1, int(ctrl[prev])))
    runs = [r for r in runs if r[1] - r[0] >= min_run]

    # group consecutive touches by the same team, keep only groups that really won the ball
    groups = []
    for a, b, team in runs:
        if groups and groups[-1][2] == team:
            groups[-1][1] = b
            groups[-1][3] += b - a
            groups[-1][4] += 1
        else:
            groups.append([a, b, team, b - a, 1])
    won = [g for g in groups if g[3] >= min_keep or g[4] >= 2]
    # after dropping a lost touch, neighbours of the same team join up again
    runs = []
    for a, b, team, _, _ in won:
        if runs and runs[-1][2] == team:
            runs[-1] = (runs[-1][0], b, team)
        else:
            runs.append((a, b, team))

    # merge neighbouring runs of the same team, so the ownership spans the gap between them
    current = None
    for k, (a, b, team) in enumerate(runs):
        nxt = runs[k + 1][0] if k + 1 < len(runs) else n
        s = 1.0 if team == HOME else -1.0
        side[a:nxt] = s
        age = np.arange(nxt - b)
        fresh[a:b] = 1.0
        fresh[b:nxt] = np.exp(-age / fade)

    # a stoppage resets ownership: after it, the restart decides
    side[stopped] = 0.0
    fresh[stopped] = 0.0
    edges = np.flatnonzero(np.diff(stopped.astype(np.int8)) == -1) + 1   # first live frame
    for e in edges:
        nxt = np.flatnonzero(ctrl[e:] >= 0)
        stop = e + (nxt[0] if len(nxt) else n - e)
        side[e:stop] = 0.0
        fresh[e:stop] = 0.0
    return side, fresh


def ball_play(ball, players, out, still, window=2 * FPS):
    """How the ball is being played, for frames where it is visible nearby in time.

    In a slow build up the ball is passed around between different players. At a stoppage it
    lies still, is out, or is carried by one player to the spot. Features (NaN when the ball is
    not seen in the window):
      ball speed now / mean over window / max over window, share of frames with a pass speed
      ball (> 8 m/s), number of different track slots that had the ball (within 1 m) in the
      window, distance from ball to the nearest pitch line.
    """
    L, W = PITCH
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    v = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    v[v > 40] = np.nan                                     # tracking jumps
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    holder = np.where(d.min(1) < 1.0, d.argmin(1), -1)

    k = 2 * window + 1
    vmax = np.full(n, np.nan)
    distinct = np.full(n, np.nan)
    pad_v = np.r_[np.full(window, np.nan), v, np.full(window, np.nan)]
    pad_h = np.r_[np.full(window, -1), holder, np.full(window, -1)]
    view_v = np.lib.stride_tricks.sliding_window_view(pad_v, k)
    view_h = np.lib.stride_tricks.sliding_window_view(pad_h, k)
    any_seen = np.isfinite(view_v).any(1)
    vmax[any_seen] = np.nanmax(view_v[any_seen], 1)
    # distinct holders in the window, counted with a sort per row (cheap enough at 2 s windows)
    hs = np.sort(view_h, axis=1)
    distinct = ((np.diff(hs, axis=1) != 0) & (hs[:, 1:] >= 0)).sum(1) + (hs[:, 0] >= 0)
    distinct = distinct.astype(float)
    distinct[~any_seen] = np.nan

    line = np.minimum.reduce([ball[:, 0], L - ball[:, 0], ball[:, 1], W - ball[:, 1]])
    fast = np.where(np.isnan(v), np.nan, (v > 8.0).astype(float))
    # No "ball out" or "ball still" inputs on purpose: those define the stoppages the model is
    # taught with, so they would be a shortcut. The decoder already uses them directly.
    feats = np.column_stack([
        v, rolling_mean(v, window), vmax, rolling_mean(fast, window), distinct,
        rolling_mean(line, window),
    ])
    feats[~any_seen] = np.nan
    return feats.astype(np.float32), any_seen
