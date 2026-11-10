# Who has the ball? Possession from tracking with a self supervised Viterbi decoder

**Team:** `LewanDATAski` · **Members:** Girijesh Kumar Paramasivam Rajakumar, Mohanraam Vasanthi Krishna, Mithun Ganapathy Vallinayagam · More Than a Hack 2027, Phase 1

Code: `possession/` in this repository. Submission: `evaluation/results/LewanDATAski.csv`.

---

## 1. Summary

We label every frame of the tactical video H, A or D using the automatic tracking (ATD) and the Smart Tagging only. No video, no hand labels, no external data.

The method has three layers:

1. **Evidence per frame**: who is touching the ball, what the team shapes say, what the tags say, whether the ball is out or placed, who is taking a restart, and how fast everyone is moving.
2. **Two small self supervised models** (gradient boosted trees) that fill the gaps where the ball is not tracked. One guesses the team in possession from player positions only, the other spots stoppages from player movement. Their training labels are produced by our own code from frames where the answer is visible, never by a person.
3. **A Viterbi decoder** over the states H, A, D with switching costs and minimum durations. It turns noisy per frame evidence into clean possession spells, which is what spellF1 rewards.

On the full match it produces **289 possessions** (mean 12.5 s) and **139 stoppages**, with the ball dead on **43%** of graded frames. That is in line with football norms (LaLiga ball in play is roughly 53 of about 100 minutes). The file is byte for byte reproducible.

---

## 2. Why this approach

**Not the video.** The organisers already turned the video into tracking. Running our own detector on 199,319 frames would cost days of compute and debugging for a team new to computer vision, with an uncertain gain.

**Not a supervised model.** Supervised learning needs labels. Hand labels on this match are not allowed, and the public 5 minutes may not be used for training or tuning. We have no labelled data from other games.

**So: rules from the definition, plus learning without human labels.** The challenge defines possession in words (control, loose balls stay with the last team, restarts). Wherever the tracking shows the answer clearly, we apply that definition directly. Where it does not (the ball is missing on 41% of graded frames), we train models on the clear frames and let them generalise to the unclear ones. A Viterbi decoder is the natural way to combine many weak per frame clues into a few long, clean segments.

---

## 3. Data assets

| Asset | Used? | How |
| --- | --- | --- |
| Tactical ATD (players and ball) | **Yes, main input** | Ball control, team shape, player tempo, ball out / still, restarts |
| Tactical Smart Tagging | **Yes** | Team phase votes; kickoff tags for the start of each half; goal tags for who kicks off after a goal |
| Tactical homography | No | Only needed to draw on the video, which we did not do |
| Tactical video | No | Not needed; the tracking is derived from it |
| Broadcast video, ATD, homography | No | Not synchronised to the tactical timeline; mapping it back was not worth the risk in the time available |
| Public 5 minute ground truth | **Only as a correctness check** | Confirmed the file format, frame alignment and that H means Barcelona. Never used to choose methods or settings |

**Tried and dropped**

* *"Opposite of last touch" for restarts.* It disagreed with the visible taker 68% of the time. Inspection showed many "ball out" moments are the thrower already holding the ball outside the line after the real exit happened while the ball was untracked. Dropped.
* *Kickoff time from the "both teams in their own half" formation.* It worked for kickoffs after goals but not at the start of the halves. Dropped, since the gain was a few seconds once.
* *Holding the last team through every ball gap* (our first version). We found that the tracker loses the ball mostly during stoppages (players move at about half their live speed inside gaps), so this produced far too few dead balls. Replaced by the stoppage logic below.

---

## 4. Training data

**No human labelled data of any kind is used**, from this game or any other.

The two models are trained on **pseudo labels made by our code**, which the rules allow ("pseudo labels produced by an algorithm, provided no person decided any of them"):

| Model | Positive / negative examples, all made by code | Inputs |
| --- | --- | --- |
| Team model | Frames where a home or away player clearly controls the ball (within 1 m, no opponent within 2 m) and the neighbouring frames agree: 22,333 frames | Player positions and speeds only (team centroids, spread, most advanced and deepest player, forward velocity), raw and averaged over 1 s and 4 s. **No ball inputs**, so it works when the ball is missing |
| Dead ball model (tempo) | Dead: visible stoppages (Section 5.3) and the minutes before each kickoff. Live: clear control frames outside stoppages | Player tempo only: mean and top speeds, share of players running, how aligned their directions are |
| Dead ball model (ball) | Same labels, only where the ball is seen within 2 s | Tempo plus how the ball is played: speed, pass speeds, how many different players touched it, distance to the nearest line. **No "ball out" or "ball still" inputs**, since those define the dead examples and would be a shortcut |

All predictions are **out of fold**: the match is cut into 10 blocks of time and each block is predicted by a model trained on the other nine, so no frame is scored by a model that saw it.

---

## 5. Possession rules we implemented

### 5.1 Control and change of possession

* A player **touches** the ball when it is within 1 m of him and no opponent is within 2 m.
* A team **wins** the ball when its consecutive touches (before the other team touches it) add up to at least 0.5 s, **or** when it touches the ball more than once in a row (it played on, for example intercepted and passed). This follows the organisers' proxies of keeping the ball or completing passes.
* **Ownership starts at the first touch** of the winning sequence, so the switch lands on the frame of the interception.
* **Not a change:** a single short touch, a deflection, or an aerial duel won without the team keeping the ball. These are ignored and the ball stays with the previous team.

### 5.2 Loose and contested balls

The challenge rule: a loose ball stays with the last team that had it. We apply it directly. The last team to win the ball keeps it at full strength while touching it; the evidence then fades over 2 s. Only after that do team shape and tags take over, typically deep inside a long ball gap.

### 5.3 Dead ball

**Stoppages we can see**

* The ball is out of the pitch for at least 8 frames, or lies still for 2 s. A placed ball is dead back to when it was last moving, up to 15 s.
* The stoppage lasts until the **restart**: a clear kick or throw (ball faster than 4 m/s for 3 frames in a row). A restart is **never taken from behind a goal line** (only throw ins come from outside the pitch, over a touchline). So a keeper walking with the ball, or the ball being rolled back onto the pitch, does not end a goal kick stoppage.
* If the tracker loses the ball during a stoppage and finds it again lying still, the whole stretch is one stoppage.
* Safety brake: after 10 s, a stoppage also ends if players are moving at live play pace.
* Every half is dead until its kickoff, found from the "Kick Off (Start)" tag and the first ball movement inside it.

**Stoppages we cannot see** are found by the dead ball models. Where both models have a view, their log odds are averaged. When they disagree (everyone walking while one player holds the ball, which could be a keeper after a save or a player waiting to take a free kick) neither can be verified, so the decoder decides from the surrounding frames.

### 5.4 Restarts

After a stoppage the restarting team has the ball:

* **Restart visible:** the taker is the player within 2.5 m of the ball just before it moves. It agreed with the team in control afterwards 95% of the time.
* **Restart hidden** (ball untracked): during a slow moment one team has a player standing just off the pitch (a throw in) or at a corner flag. That team takes it. Agreement about 80%.
* **After a goal:** the team that did not score kicks off (scorer from the GOALS tag).

### 5.5 Durations

A possession lasts at least 1 s (real control). A stoppage lasts at least 4 s (roughly the quickest throw in).

---

## 6. How the evidence is combined

For every frame we build two log odds: **home vs away** and **dead vs alive**.

* Home vs away = last controlling team × freshness + (1 − freshness) × (team model + Smart Tagging vote) + restart taker votes.
* Dead vs alive = dead ball model + a small push when the ball is missing; forced dead inside visible stoppages and before kickoff; forced alive for 1 s after a visible restart.

These become per frame scores for H, A and D, counted per second of evidence. A Viterbi decoder (one per half) finds the best path. Switching team costs 1, entering or leaving a stoppage costs 3, and minimum durations are enforced with an expanded state chain (each label has sub states counting frames up to its minimum).

Tag codes naming the attacking team (Build Up, Progression, Attacking Transition, Ball in Final Third, Creating Chances, Long Ball, Ball in the Box, Players in the Box, Goals, Set Pieces) vote for that team. Defending codes vote for the other team. We chose the direction of each code by checking it against our own control frames, not the ground truth.

**How settings were chosen.** The ranking window may not be used for tuning, so every setting comes from one of three places:

1. **Football facts**: minimum durations, the typical number of stoppages and possessions, ball in play time.
2. **Measured reliability against our own code made labels**: restart clue weights are the log odds of how often each clue was right (95% → 2.9, 80% → 1.4).
3. **Held out self checks**: we hide the control evidence in alternate minutes and check whether the decoder still finds the team in control and the right restarting team.

---

## 7. Self checks (no ground truth used)

| Check | Result |
| --- | --- |
| Team model, out of fold, against control frames | 84% |
| Clear visible turnovers (26) where the decoded switch lands within 3 frames of the first control by the new team | 75% (24 of 26 found) |
| Restarting team right with control evidence hidden (two splits) | 75% / 81% |
| Possessions / mean length | 289 / 12.5 s |
| Stoppages / dead share | 139 / 43% |

Version history, each step justified by the checks above:

| | v1 | v2 | v3 | v4 | v5 (submitted) |
| --- | --- | --- | --- | --- | --- |
| Main change | Decoder + team model | Visible stoppages, tempo dead model | Restart takers | Last team rule, real control, ball aware dead model | Restart = clear kick, no shortcuts, models count equally |
| Possessions | 284 | 296 | 295 | 331 | 289 |
| Stoppages | 103 | 171 | 171 | 166 | 139 |
| Turnovers caught (of 26) | | | 7 | 25 | 24 |

Correctness check on the public 5 minutes (run after each version was chosen, never used to choose): the labels align, H is Barcelona, and team and dead ball labels are clearly better than the baseline. With 17 possessions it cannot rank versions, as EVALUATION.md demonstrates.

---

## 8. Limitations and what we would do with more time

* **Long ball gaps.** Inside a gap the team comes from player shape and tags, which is right about 80% of the time when we hide the ball. Turnovers that happen entirely inside a gap are found late or missed.
* **Exact restart time inside gaps.** When the restart happens while the ball is untracked, the start of the next possession comes from the tempo model and can be seconds off, which costs spellF1.
* **Keeper with the ball vs set piece waiting.** Our two dead ball models disagree there, and nothing in the tracking settles it.
* **Fouls and offsides.** There is no whistle in the data. We only see them as a drop in tempo or a placed ball.
* **Control is a distance rule.** Tracking noise and players near the ball in duels create false touches. The 0.5 s rule removes most of them but not all.

**With more time:** recover the ball inside gaps from the video with a ball detector, starting from the tracked positions and the homography; use the broadcast feed to cover tactical camera gaps; and learn the possession rules from hand labelled data of other matches, which the rules allow.

---

## 9. Reproduction

Requirements: Python 3.13 and the pinned packages in `requirements-possession.txt`.

```bash
pip install -r requirements-possession.txt
# point to the data bundle (a folder that contains tactical/)
export MTAH_DATA_DIR=/path/to/data            # macOS / Linux
$env:MTAH_DATA_DIR = "C:\path\to\data"        # Windows PowerShell

python -m possession.run --fresh --name LewanDATAski
python evaluation/validate_submission.py evaluation/results/LewanDATAski.csv
```

A fresh run takes about 2 minutes (longer on a slow laptop), then caches its evidence in `data/` (rerun with `--fresh` to rebuild). Two fresh runs give a byte identical file. All settings are in `PARAMS` at the top of `possession/run.py`. The match data is not in this repository.

| File | Role |
| --- | --- |
| `possession/load.py` | Read ATD and Smart Tagging into arrays (cached) |
| `possession/features.py` | Ball control, team shape and tempo, stoppages, restarts, last team rule |
| `possession/models.py` | The two self supervised models, out of fold |
| `possession/decode.py` | Viterbi decoder with minimum durations |
| `possession/run.py` | Pipeline and settings |
