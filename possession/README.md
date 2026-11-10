# Our possession model (v4)

## How to run

Your match data lives in `C:\GK\Chalmers\Barcathon\tactical`, so point the code there first (PowerShell):

```
$env:MTAH_DATA_DIR = "C:\GK\Chalmers\Barcathon"
pip install -r requirements-possession.txt
python -m possession.run
python evaluation/validate_submission.py evaluation/results/barcathon_gk.csv
```

First run takes about 2 minutes. After that the evidence is cached in `data/` and a run takes seconds.
Use `--fresh` to rebuild from scratch.

## The idea in plain words

1. **Who has the ball when we can see it.** A player within 1 m of the ball, with no opponent within 2 m, is touching it. A team WINS the ball when its touches add up to half a second, or it touches it more than once in a row (plays on). A single short touch, like a header won but not kept, does not count. Then the challenge's own rule: a loose ball stays with the last team that won it. This evidence leads; it fades over 2 s after the last touch, and only then do team shape and tags take over.
2. **Who has the ball when we can't.** A small model learns to tell "Barça attacking" from "Athletic attacking" using only player positions and speeds (who is pushed up, how spread out each team is). Its training labels come from step 1, made by code, never by hand.
3. **Smart Tagging hints.** Build Up, Progression etc. name the team with the ball. Defending tags name the team without it.
4. **Dead ball, the stoppages we can see.** The ball goes out of the pitch, or lies still before a set piece. We mark it dead from that moment until the restart (the first frame the ball moves again inside the pitch). If the tracker loses the ball in between and finds it again lying still, the whole stretch is one stoppage.
5. **Dead ball, the ones we can't see.** A second small model learns what a stoppage looks like from player TEMPO only (speeds, nobody sprinting, no shared direction). It trains on the stoppages from step 4, so it can spot stoppages while the tracker has lost the ball. Where the ball is visible within 2 s, a version that also sees how the ball is played (passes, how many different players touched it) takes over, so a slow build up with everyone walking is not mistaken for a stoppage.
6. **Who restarts.** Whoever takes the restart has the ball after it. If the restart is visible, the taker is the player on the ball (right about 95% of the time in our own check). If the tracker lost the ball, the taker still shows up in the players: during a slow moment one team has a player standing just off the pitch (throw in) or at a corner flag (about 80%). After a goal tag, the other team kicks off. Each clue is weighted by the log odds of how often it was right.
7. **Viterbi decoder.** Each frame scores H, A and D. Switching costs a penalty, a possession must last at least 1 s and a stoppage at least 4 s (about the quickest throw in). This gives clean possession chunks, which is what spellF1 rewards.

## Sanity numbers (full match, no answer key used)

| | v1 | v2 | v3 | v4 |
| --- | --- | --- | --- | --- |
| Possessions | 284 | 296 | 295 | 331 |
| Mean possession | 17.9 s | 12.7 s | 12.7 s | 11.6 s |
| Dead ball share | 20% | 41% | 41% | 40% (LaLiga ball in play is about 53 of ~100 min) |
| Clear turnovers caught (of 26) | | | 7 | 25, 76% within 3 frames |
| Clear control frames called dead | | 16% | 16% | 7% |
| Restart team right, ball control hidden | | 74% / 80% | 77% / 83% | 73% / 78% (more restarts counted, not like for like) |

The last row hides the "player has the ball" evidence in alternate minutes and checks the team picked after each stoppage against the team our own code sees controlling the ball soon after.

## Files

| File | What it does |
| --- | --- |
| `load.py` | Reads tracking and tags into arrays |
| `features.py` | Ball control, team shape, tag votes, ball out / still |
| `models.py` | The two self supervised models (out of fold) |
| `decode.py` | Viterbi decoder |
| `run.py` | Puts it together, all settings in `PARAMS` |

## Rules we follow

No hand labels. Settings are chosen from football common sense and a held out check on our own code made labels, never tuned on the public 5 minutes.

## Reproducible

Two fresh runs give a byte identical CSV (fixed seeds, no randomness elsewhere).
