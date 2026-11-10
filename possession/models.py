"""Two small self supervised models.

Neither model sees a single human label. Their training labels are made by code:

* team model  : frames where one player clearly controls the ball give "home" or "away".
                The model learns to tell the two apart from the PLAYERS ONLY (team shape,
                speed, who is pushed up), so it still works when the ball is missing.
* dead model  : whole stoppages we can see (ball out until the restart, ball placed still,
                before kickoff) give "dead"; clear control frames give "live". The model learns
                what a stoppage looks like from the players, so it can spot one when the
                tracker has lost the ball.

Predictions are out of fold: the match is cut into blocks of time, and each block is predicted by
a model trained on the other blocks. So a frame is never scored by a model that trained on it.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import HALVES, rolling_mean

N_FOLDS = 10
SUBSAMPLE = 5   # neighbouring frames are near copies, so every 5th frame is plenty
SEED = 0


def _graded_index(n):
    idx = np.concatenate([np.arange(a, b) for a, b in HALVES])
    return idx[idx < n]


def out_of_fold(X, y, weight=None):
    """P(y == 1) for every graded frame, out of fold. y is 1, 0 or -1 (no label)."""
    n = len(X)
    graded = _graded_index(n)
    folds = np.array_split(graded, N_FOLDS)
    prob = np.full(n, np.nan)
    for k, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != k])
        train = train[y[train] >= 0][::SUBSAMPLE]
        w = None if weight is None else weight[train]
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                               max_leaf_nodes=31, l2_regularization=1.0,
                                               random_state=SEED)
        model.fit(X[train], y[train], sample_weight=w)
        prob[test] = model.predict_proba(X[test])[:, 1]
    return prob


def team_pseudo_labels(ctrl, out):
    """1 = home has it, 0 = away, -1 = unknown. Only frames whose neighbours agree."""
    h = rolling_mean(np.where(ctrl >= 0, (ctrl == 0).astype(float), np.nan), 12)
    y = np.full(len(ctrl), -1, dtype=np.int8)
    y[(ctrl == 0) & (h > 0.8)] = 1
    y[(ctrl == 1) & (h < 0.2)] = 0
    y[out] = -1
    return y


def dead_pseudo_labels(team_y, stopped, before_kickoff):
    """1 = dead ball (a whole stoppage we can see), 0 = live (clear control), -1 = unknown."""
    y = np.full(len(team_y), -1, dtype=np.int8)
    y[team_y >= 0] = 0
    y[stopped | before_kickoff] = 1
    return y


def balance(y):
    """Sample weights so both classes count the same."""
    w = np.zeros(len(y))
    for c in (0, 1):
        m = y == c
        if m.any():
            w[m] = 0.5 / m.mean()
    return w
