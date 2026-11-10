# Who has the ball? Team `LewanDATAski`, More Than a Hack 2027

Our entry to Phase 1 of [More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027):
label every frame of Barcelona vs Athletic Club as home possession (H), away possession (A) or dead
ball (D).

**Approach in one line:** the challenge's possession rules applied to the tracking where the answer
is visible, two small self supervised models (trained only on labels made by our own code) to fill
the gaps where the ball is lost, and a Viterbi decoder that turns it all into clean possession spells.

* Full method, rules, assets, limitations: **[LewanDATAski.md](LewanDATAski.md)**
* Code: **[possession/](possession/)** (step by step notes in [possession/README.md](possession/README.md))
* No hand labels of any kind, and no match data in this repository.

## Reproduce the submission

1. Get the match data from the organisers (see [DATA.md](DATA.md)) and unpack it so you have a folder
   containing `tactical/` (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`,
   `BAR-ATH_tactical_smart_tagging.xml`). Either put it in `data/` here or point to it:

   ```bash
   export MTAH_DATA_DIR=/path/to/folder            # macOS / Linux
   $env:MTAH_DATA_DIR = "C:\path\to\folder"        # Windows PowerShell
   ```

2. Install the pinned packages (Python 3.13) and run:

   ```bash
   pip install -r requirements-possession.txt
   python -m possession.run --fresh --name LewanDATAski
   python evaluation/validate_submission.py evaluation/results/LewanDATAski.csv
   ```

   About 2 minutes. Two fresh runs give a byte identical file.

## What is in here

| Path | What |
| --- | --- |
| `possession/` | Our method |
| `LewanDATAski.md` | The technical document |
| `tools/check_before_push.py` | Safety check run before every upload: no match data may be published |
| `games.py`, `evaluation/`, `starter/`, `*.md` | The organisers' starter kit, unchanged (MIT, see [LICENSE](LICENSE)); their original README is [CHALLENGE_README.md](CHALLENGE_README.md) |

## Data and licence

The match data (videos, tracking, Smart Tagging, homography) is licensed by the organisers for this
competition only and is **not** in this repository. `.gitignore` blocks it by folder, name and file
type, and `tools/check_before_push.py` checks every upload. The code is MIT licensed; the starter
kit is © Metrica Sports, see [LICENSE](LICENSE).
