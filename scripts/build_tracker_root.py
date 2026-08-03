"""Write a companion ROOT file of sentinel-filled tracker positions, one row
per ROOT event, for one or more runs.

Packages whatever alignment ``align_tracker_to_root_by_timestamp`` produces
into a small, reusable file: same event count/order as the run's own
``EventTree``, so it can be read alongside it by matching row index (or
used as a ROOT "friend tree"). A ROOT event with no trustworthy tracker
match gets the tracker's own existing no-hit sentinels (-1000/-2000/-3000/
-4000 for x1/y1/x2/y2) instead of being silently dropped, so downstream
code just needs ``!= -1000`` (etc) rather than re-running the whole
alignment pipeline (spill-boundary finding, per-segment offset search,
loading the raw .dat files) every time it wants tracker positions.

This doesn't add any accuracy of its own -- see
align_tracker_to_root_by_timestamp's docstring for how the alignment
itself works (matching by real clock time, not by counting events) and
its known precision limits. It just gives whatever that function produces
a stable, reusable form instead of recomputing it on every analysis run.

Usage:
    python -m scripts.build_tracker_root --run 1832
    python -m scripts.build_tracker_root --run 1832 1774 1817
    python -m scripts.build_tracker_root --all
"""
import argparse
import os

import numpy as np
import uproot

from utils.data import get_run_filepath, load_run_list
from utils.io import ensure_output_dir
from utils.tracker import build_aligned_tracker_branches, load_tracker_run

OUTPUT_DIR = ensure_output_dir("aligned_tracker")


def build_one_run(run_id):
    filepath = get_run_filepath(run_id)
    tracker = load_tracker_run(run_id)
    with uproot.open(filepath) as f:
        tree = f["EventTree"]
        trigger_n = tree["trigger_n"].array(library="np")
        root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")

    branches, match_frac = build_aligned_tracker_branches(tracker, trigger_n, root_tstamp)

    out_path = os.path.join(OUTPUT_DIR, f"{run_id}_tracker.root")
    with uproot.recreate(out_path) as fout:
        fout.mktree("TrackerAligned", {name: arr.dtype for name, arr in branches.items()})
        fout["TrackerAligned"].extend(branches)

    n_matched = int(branches["tracker_matched"].sum())
    n_root = len(trigger_n)
    print(f"[{run_id}] {n_matched}/{n_root} ROOT events got a tracker match "
          f"({n_matched/n_root:.1%}, tracker-side match_frac={match_frac:.1%}) "
          f"-> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", help="Run ID(s) to process")
    parser.add_argument("--all", action="store_true", help="Process every run in run_list.json")
    args = parser.parse_args()

    if args.all:
        runs = list(load_run_list().keys())
    elif args.run:
        runs = args.run
    else:
        parser.error("pass --run <id> [<id> ...] or --all")

    for run_id in runs:
        try:
            build_one_run(run_id)
        except Exception as e:
            print(f"[{run_id}] skipped: {e}")


if __name__ == "__main__":
    main()
