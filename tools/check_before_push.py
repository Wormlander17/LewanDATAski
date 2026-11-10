"""Run this before every upload to GitHub. It stops you if match data would be published.

    python tools/check_before_push.py

It lists every file git would publish (tracked files plus new files not ignored) and fails if any
of them looks like match data: a data/ file other than its README, a match file name, a cache,
a video or archive, a file starting like the tracking text, or anything over 2 MB.

The match data is licensed for this competition only. Publishing it breaks the terms you signed.
Never upload by dragging files onto the GitHub website: that ignores .gitignore completely.
"""

import fnmatch
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
    if "--no-git" in sys.argv:
        return None
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def read_ignore_rules():
    """The patterns in .gitignore as (negated, directory_only, anchored, pattern)."""
    rules = []
    path = os.path.join(ROOT, ".gitignore")
    if not os.path.exists(path):
        return rules
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip()
            if not line or line.startswith("#"):
                continue
            neg = line.startswith("!")
            if neg:
                line = line[1:]
            dir_only = line.endswith("/")
            line = line.rstrip("/")
            anchored = "/" in line
            rules.append((neg, dir_only, anchored, line.lstrip("/")))
    return rules


def _match(rule, rel, is_dir):
    _, dir_only, anchored, pat = rule
    if dir_only and not is_dir:
        return False
    if anchored:
        return fnmatch.fnmatchcase(rel, pat)
    return fnmatch.fnmatchcase(rel.rsplit("/", 1)[-1], pat)


def ignored(rules, rel, is_dir):
    """Same idea as git: the last matching rule wins."""
    result = False
    for rule in rules:
        if _match(rule, rel, is_dir):
            result = not rule[0]
    return result


def all_files():
    """Files git would publish, worked out from .gitignore without git (for GitHub Desktop)."""
    rules = read_ignore_rules()
    out = []
    for folder, dirs, files in os.walk(ROOT):
        rel_folder = os.path.relpath(folder, ROOT).replace(os.sep, "/")
        rel_folder = "" if rel_folder == "." else rel_folder + "/"
        keep = []
        for d in dirs:
            if d == ".git" or ignored(rules, rel_folder + d, True):
                continue
            keep.append(d)
        dirs[:] = keep
        for name in files:
            rel = rel_folder + name
            # a file inside an ignored folder is ignored too (git never looks inside it)
            if not ignored(rules, rel, False):
                out.append(rel)
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
        print("(git not found in this window, so the .gitignore rules are applied by this script)")
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
