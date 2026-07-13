"""SiPM HV (bias voltage) vs. run number, for the hodoscope's two HG readout boards.

Each event carries a directly-monitored bias voltage for every FERS board
(``FERS_Board<N>_SipmHV``); for the hodoscope that's board 0 (Y plane) and
board 1 (X plane). This averages that per-event value over each run and
plots it against run number -- a quick way to spot HV setting changes,
drift, or instability across the dataset.

Runs are loaded in parallel (one process per run, sized to the node's
available CPUs), same approach as sieff_scan.py.

Usage:
    python scripts/hgvoltage_scan.py
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:
        """No-op stand-in so the scan still runs without the optional dependency."""
        def __init__(self, *a, total=None, **k):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        write = staticmethod(print)

from utils.data import get_run_filepath, load_run_list
from utils.io import ensure_output_dir
from utils.plotting import get_runtype

OUTPUT_DIR = ensure_output_dir("hgvoltage_scan")

# Board0 reads out the hodoscope's Y plane, Board1 the X plane (matches the
# Y_HG/X_HG convention used throughout scripts/sieff.py, effplots.py, etc.).
HV_BRANCHES = {"hodo_y": "FERS_Board0_SipmHV", "hodo_x": "FERS_Board1_SipmHV"}
RUNTYPE_COLORS = {"TB2025": "C0", "Cosmic 2025": "gray", "TB2026": "C1"}

N_WORKERS = len(os.sched_getaffinity(0))


def _load_one(run_id):
    """Picklable per-run worker: per-event SiPM HV, both hodoscope boards."""
    try:
        filepath = get_run_filepath(run_id)
        paths = filepath if isinstance(filepath, list) else [filepath]
        hv_x, hv_y = [], []
        for path in paths:
            with uproot.open(path) as f:
                tree = f["EventTree"]
                hv_x.append(tree[HV_BRANCHES["hodo_x"]].array(library="np"))
                hv_y.append(tree[HV_BRANCHES["hodo_y"]].array(library="np"))
        return run_id, (np.concatenate(hv_x), np.concatenate(hv_y)), None
    except Exception as e:
        return run_id, None, str(e)


def load_all_runs(run_ids):
    """Load every run in ``run_ids`` in parallel. Returns {run_id: (hv_x, hv_y)}, skipping failures."""
    results = {}
    with tqdm(total=len(run_ids), desc="Scanning runs", unit="run") as pbar, \
        ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_one, run_id): run_id for run_id in run_ids}
        for fut in as_completed(futures):
            run_id, result, err = fut.result()
            pbar.set_postfix_str(f"run {run_id}")
            if err is not None:
                tqdm.write(f"    [skip] run {run_id}: {err}")
            else:
                results[run_id] = result
            pbar.update(1)
    return results


def summarize(hv):
    """Mean +/- std HV over the run (std should be tiny -- it's a monitored setting, not a signal)."""
    return float(np.mean(hv)), float(np.std(hv))


def plot_hv_vs_run(rows, filename):
    if not rows:
        print("No rows to plot.")
        return
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 8))

    runtypes = sorted({r["runtype"] for r in rows})
    for runtype in runtypes:
        sub = [r for r in rows if r["runtype"] == runtype]
        run_ids = [r["run_id"] for r in sub]
        color = RUNTYPE_COLORS.get(runtype, "black")
        ax.errorbar(run_ids, [r["hodo_x_hv"] for r in sub], yerr=[r["hodo_x_hv_std"] for r in sub],
                   marker="o", ls="none", ms=4, color=color, alpha=0.9,
                   label=f"Hodo X (Board1) -- {runtype}")
        ax.errorbar(run_ids, [r["hodo_y_hv"] for r in sub], yerr=[r["hodo_y_hv_std"] for r in sub],
                   marker="^", ls="none", ms=4, color=color, alpha=0.4,
                   label=f"Hodo Y (Board0) -- {runtype}")

    ax.set_xlabel("Run number", loc="right")
    ax.set_ylabel("SiPM HV [V]", loc="top")
    ax.legend(fontsize=13, ncol=2)
    mh.label.exp_label(exp="CaloX", data=True, rlabel="Hodoscope SiPM HV vs Run", ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"HV-vs-run plot saved {filename}")
    plt.close()


def write_summary(rows, filename):
    if not rows:
        print(f"No rows to write to {filename}.")
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {filename}")


def main():
    run_ids = sorted(load_run_list().keys(), key=int)
    print(f"Loading {len(run_ids)} run(s) using {N_WORKERS} worker process(es)...")
    results = load_all_runs(run_ids)

    rows = []
    for run_id in run_ids:
        if run_id not in results:
            continue
        hv_x, hv_y = results[run_id]
        x_mean, x_std = summarize(hv_x)
        y_mean, y_std = summarize(hv_y)
        rows.append({
            "run_id": int(run_id), "runtype": get_runtype(run_id), "n_events": len(hv_x),
            "hodo_x_hv": x_mean, "hodo_x_hv_std": x_std,
            "hodo_y_hv": y_mean, "hodo_y_hv_std": y_std,
        })

    plot_hv_vs_run(rows, os.path.join(OUTPUT_DIR, "hv_vs_run.png"))
    write_summary(rows, os.path.join(OUTPUT_DIR, "hv_vs_run.csv"))


if __name__ == "__main__":
    main()
