"""Run this before every upload to GitHub. It stops you if match data would be published.

    python tools/check_before_push.py

It lists every file git would publish (tracked files plus new files not ignored) and fails if any
of them looks like match data: a data/ file other than its README, a match file name, a cache,
a video or archive, a file starting like the tracking text, or anything over 2 MB.

The match data is licensed for this competition only. Publishing it breaks the terms you signed.
Never upload by dragging files onto the GitHub website: that ignores .gitignore completely.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 2_000_000          # the public ground truth and baselines are about 1.7 MB each
ALLOWED_BIG = {
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",
}
BAD_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".npz", ".npy", ".pkl", ".parquet",
                ".h5", ".zip", ".tar", ".gz", ".7z", ".rar", ".pdf")
BAD_NAMES = ("_atd.txt", "_atd.xml", "_smart_tagging.xml", "_homography.csv")
ALLOWED_DATA = {"data/README.md"}


def files_git_would_publish():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def all_files():
    out = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in files:
            out.append(os.path.relpath(os.path.join(folder, name), ROOT).replace(os.sep, "/"))
    return out


def looks_like_tracking(path):
    """The ATD text starts like '1:0.33411,0.90565,0.88;NaN,NaN,NaN;...'."""
    try:
        with open(os.path.join(ROOT, path), "rb") as f:
            head = f.read(200).decode("utf-8", "ignore")
    except OSError:
        return False
    first = head.split("\n", 1)[0]
    return first.count(";") > 5 and ":" in first and "," in first


def problems(path):
    p = path.replace("\\", "/")
    low = p.lower()
    found = []
    if (low.startswith("data/") or "/data/" in low) and p not in ALLOWED_DATA:
        found.append("inside a data folder")
    if low.startswith(("tactical/", "broadcast/")) or "/tactical/" in low or "/broadcast/" in low:
        found.append("inside a tactical/broadcast folder")
    if low.endswith(BAD_SUFFIXES):
        found.append("video, cache, archive or pdf file type")
    if any(low.endswith(b) for b in BAD_NAMES):
        found.append("match data file name")
    full = os.path.join(ROOT, p)
    if os.path.isfile(full):
        size = os.path.getsize(full)
        if size > MAX_BYTES and p not in ALLOWED_BIG and not p.startswith("evaluation/results/baseline_"):
            found.append("large file ({:.1f} MB)".format(size / 1e6))
        if looks_like_tracking(p):
            found.append("content looks like tracking data")
    if p.startswith("evaluation/results/") and p.endswith(".csv") and "baseline_" not in p:
        found.append("a submission file (keep it private until the deadline, upload it via the form)")
    return found


def main():
    files = files_git_would_publish()
    if files is None:
        print("git was not found, so this checks EVERY file in the folder (stricter than needed).")
        print("With GitHub Desktop, open its menu Repository > Open in Command Prompt and run this")
        print("script there, so it can see what git would actually publish.\n")
        files = all_files()

    bad = [(f, problems(f)) for f in files]
    bad = [(f, p) for f, p in bad if p]
    print("{} files would be published.".format(len(files)))
    if bad:
        print("\nSTOP. Do not upload. These files must not go public:\n")
        for f, p in bad:
            print("  {}  <- {}".format(f, ", ".join(p)))
        print("\nFix: delete them from this folder, or add them to .gitignore. If git already")
        print("tracks one, run:  git rm --cached \"<file>\"  and commit again.")
        sys.exit(1)
    print("OK. No match data, caches or submission files would be published.")


if __name__ == "__main__":
    main()
