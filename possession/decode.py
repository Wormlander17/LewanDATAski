"""Viterbi decoding: pick the smoothest H / A / D path that fits the evidence.

Each frame gives a score for each state. Changing state costs a fixed penalty. The decoder finds
the path with the best total score, so it only switches team when the evidence for the new team
is strong enough, for long enough, to pay the penalty. That gives clean possession chunks
instead of flicker, which is what spellF1 rewards.
"""

import numpy as np

H, A, D = 0, 1, 2


def viterbi(emis, switch_team, switch_dead):
    """emis: (T, 3) log scores per frame. Returns the best state per frame."""
    T = len(emis)
    trans = np.array([
        [0.0, -switch_team, -switch_dead],
        [-switch_team, 0.0, -switch_dead],
        [-switch_dead, -switch_dead, 0.0],
    ])
    back = np.zeros((T, 3), dtype=np.int8)
    score = emis[0].copy()
    for t in range(1, T):
        cand = score[:, None] + trans          # from state (row) to state (col)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + emis[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_duration(emis, switch_team, switch_dead, min_len):
    """Viterbi where each state must last at least min_len[state] frames.

    Each label gets a chain of sub states that count frames up to its minimum. The label can only
    be left from the last sub state, so nothing shorter than the minimum can appear.
    """
    T = len(emis)
    m = [max(1, int(k)) for k in min_len]
    cost = np.array([
        [0.0, switch_team, switch_dead],
        [switch_team, 0.0, switch_dead],
        [switch_dead, switch_dead, 0.0],
    ])
    neg = -1e18
    S = [np.full(m[l], neg) for l in range(3)]
    for l in range(3):
        S[l][-1] = emis[0, l]           # a half may start in any state
    back0 = np.zeros((T, 3), dtype=np.int8)       # which label entered sub state 0
    backlast = np.zeros((T, 3), dtype=bool)       # last sub state came from itself
    for t in range(1, T):
        ends = np.array([S[l][-1] for l in range(3)])
        new = []
        for l in range(3):
            enter = ends - cost[:, l]
            enter[l] = neg
            src = int(np.argmax(enter))
            back0[t, l] = src
            e = emis[t, l]
            s = np.empty(m[l])
            if m[l] == 1:
                stay = S[l][0]
                backlast[t, l] = stay >= enter[src]
                s[0] = max(stay, enter[src]) + e
            else:
                s[0] = enter[src] + e
                s[1:] = S[l][:-1] + e
                if S[l][-1] >= S[l][-2]:
                    s[-1] = S[l][-1] + e
                    backlast[t, l] = True
            new.append(s)
        S = new
    # backtrack
    l = int(np.argmax([S[k][-1] for k in range(3)]))
    i = m[l] - 1
    path = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = l
        if t == 0:
            break
        if m[l] == 1:
            if not backlast[t, l]:
                l = int(back0[t, l]); i = m[l] - 1
        elif i == 0:
            l = int(back0[t, l]); i = m[l] - 1
        elif i == m[l] - 1 and backlast[t, l]:
            pass
        else:
            i -= 1
    return path


def emissions(team_logodds, dead_logodds, scale):
    """Per frame log probabilities of H, A, D from two log odds, scaled to per second units."""
    log_sig = lambda z: -np.logaddexp(0.0, -z)   # log of the logistic function
    alive = log_sig(-dead_logodds)
    e = np.column_stack([
        alive + log_sig(team_logodds),
        alive + log_sig(-team_logodds),
        log_sig(dead_logodds),
    ])
    return e * scale
