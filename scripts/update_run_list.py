"""Rebuild data/run_list.json from yofeng's converted-ROOT directory plus a
beam-conditions CSV (run number -> beam type / energy).

Each run entry becomes ``{"file": <path or [paths]>, "beam_type": ..., "beam_energy_gev": ...}``.
Beam fields are ``null`` for runs outside the CSV's coverage (e.g. TB2025).

Usage:
    python scripts/update_run_list.py [--csv data/TB2026_beam_conditions.csv]
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

from utils.io import project_root

YOFENG_ROOT_DIR = "/lustre/research/hep/yofeng/HG-DREAM/CERN/ROOT/"
RUN_LIST_PATH = project_root() / "data" / "run_list.json"
DEFAULT_CSV_PATH = project_root() / "data" / "TB2026_beam_conditions.csv"

FILENAME_RE = re.compile(r"^run(\d+)_(\d+)(?:_converted)?\.root$")


def scan_available_runs(root_dir=YOFENG_ROOT_DIR):
    """Return {run_id (int): [filepath, ...]} sorted chronologically by the filename timestamp."""
    by_run = defaultdict(list)
    for fname in os.listdir(root_dir):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        run_id, timestamp = int(m.group(1)), m.group(2)
        by_run[run_id].append((timestamp, os.path.join(root_dir, fname)))
    return {run_id: [p for _, p in sorted(paths)] for run_id, paths in by_run.items()}


def _parse_beam_energy(raw):
    raw = (raw or "").strip()
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return None


def load_beam_conditions(csv_path):
    """Return {run_id (int): (beam_type, beam_energy_gev)} parsed from the elog CSV."""
    beam = {}
    with open(csv_path, newline="") as f:
        first_line = f.readline()
        if not first_line.startswith("RunNumber"):
            f.seek(0)
            f.readline()  # skip leading blank row before the real header
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            run_raw = (row.get("RunNumber") or "").strip()
            if not run_raw.isdigit():
                continue
            run_id = int(run_raw)
            beam_type = (row.get("beam type") or "").strip() or None
            if beam_type in ("0", "NA"):
                beam_type = None
            beam_energy = _parse_beam_energy(row.get("beam energy [GeV]"))
            beam[run_id] = (beam_type, beam_energy)
    return beam


def build_run_list(available, beam_conditions, existing):
    """Merge filesystem availability + beam metadata into the new run_list.json schema.

    ``existing`` entries (old schema: run_id -> path or [paths]) are kept as
    the file listing for runs that are no longer visible on disk, but the
    filesystem scan always wins when a run *is* visible (it can pick up
    files added since the entry was written, e.g. multi-file re-conversions).
    """
    merged = {}
    all_ids = set(available) | {int(k) for k in existing}
    for run_id in sorted(all_ids):
        if run_id in available:
            files = available[run_id]
        else:
            old = existing[str(run_id)]
            files = old if isinstance(old, list) else [old]
        beam_type, beam_energy_gev = beam_conditions.get(run_id, (None, None))
        merged[str(run_id)] = {
            "file": files[0] if len(files) == 1 else files,
            "beam_type": beam_type,
            "beam_energy_gev": beam_energy_gev,
        }
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Beam-conditions CSV path")
    parser.add_argument("--root-dir", default=YOFENG_ROOT_DIR, help="yofeng converted-ROOT directory")
    args = parser.parse_args()

    with open(RUN_LIST_PATH) as f:
        existing = json.load(f)

    available = scan_available_runs(args.root_dir)
    beam_conditions = load_beam_conditions(args.csv)
    merged = build_run_list(available, beam_conditions, existing)

    n_new = len(merged) - len(existing)
    n_with_beam = sum(1 for e in merged.values() if e["beam_type"] is not None)
    print(f"Runs: {len(existing)} -> {len(merged)} ({n_new:+d})")
    print(f"Runs with beam metadata: {n_with_beam}/{len(merged)}")

    with open(RUN_LIST_PATH, "w") as f:
        json.dump(merged, f, indent=4)
        f.write("\n")
    print(f"Wrote {RUN_LIST_PATH}")


if __name__ == "__main__":
    main()
