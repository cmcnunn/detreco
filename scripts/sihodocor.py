import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot
import mplhep as mh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.data import get_run_filepath
from utils.tracker import (
    load_tracker_run,
    station1_hit_mask,
    station2_hit_mask,
    align_tracker_to_root_by_timestamp,
    calibrate_tracker_positions,
    TRACKER_HODO_MIN_FIT_R,
    TRACKER_HODO_MIN_FIT_POINTS,
)
from utils.plotting import get_beam_label, get_runs_by_testbeam, draw_fit, fit_profile_line, _hist_edges
from utils.constants import X_MAPPING, Y_MAPPING
from utils.hodo import reconstruct_hodoscope
from utils.selectors import get_branch_names, passes_veto, get_counter_branch_names, counter_1cm_hit_mask, counter_3cm_hit_mask

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:
        """No-op stand-in so the script still runs without the optional dependency."""
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

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/sihodocor"
RUNS_OUTPUTDIR = os.path.join(OUTPUTDIR, "runs")
BIGPLOT_OUTPUTDIR = os.path.join(OUTPUTDIR, "bigplot")
os.makedirs(RUNS_OUTPUTDIR, exist_ok=True)
os.makedirs(BIGPLOT_OUTPUTDIR, exist_ok=True)

# One worker per available CPU (respects a SLURM/cgroup allocation, unlike
# os.cpu_count()); bump your job's cpu allocation and this picks it up
# automatically, no flag needed.
N_WORKERS = len(os.sched_getaffinity(0))

# White-on-black-outline so the fit line/annotation stay legible over every
# viridis bin color, from the dark low-count purple to the bright yellow peak.

def plot_sihodocor(xh, yh, x1, y1, run_id, trackern, selection="", runtype="", OUTPUTDIR=OUTPUTDIR):
    if runtype == "":
        runtype = get_beam_label(run_id)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(xh, x1, bins=[_hist_edges(xh), _hist_edges(x1)])
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    tag = f"Run {run_id}" + (f" -- {selection}" if selection else "")
    draw_fit(ax, xh, x1, tag=tag)
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=f"HodoX vs Tracker{trackern}", data=True)
    ax.set_xlabel("Hodo X [mm]", loc='right')
    ax.set_ylabel("Silicon Tracker X [mm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_x_{run_id}_{selection}.png"), dpi=300)
    print("Hodo vs Tracker Plot Saved " + os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_x_{run_id}_{selection}.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(yh, y1, bins=[_hist_edges(yh), _hist_edges(y1)])
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    draw_fit(ax, yh, y1, tag=tag)
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=f"HodoY vs Tracker{trackern}", data=True)
    ax.set_xlabel("Hodo Y [mm]", loc='right')
    ax.set_ylabel("Silicon Tracker Y [mm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_y_{run_id}_{selection}.png"), dpi=300)
    print("Hodo vs Tracker Plot Saved " + os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_y_{run_id}_{selection}.png"))
    plt.close(fig)


def load_run_data(run):
    """Load, align and mask one run's tracker/hodoscope/veto/counter data.

    Shared by the single-run plotting path and the ``--testbeam`` slope
    aggregation path, so both see identical alignment/masking. Returns
    ``None`` (after printing why) if the run's ROOT file can't be read.
    """
    filepath = get_run_filepath(run)
    si_data = load_tracker_run(run)
    mask1, mask2 = station1_hit_mask(si_data), station2_hit_mask(si_data)
    veto_branch, _, _ = get_branch_names(run)
    has_counters = int(run) >= 1825
    if has_counters:
        one_cm_branch, three_cm_branch = get_counter_branch_names(run)
    try:
        with uproot.open(filepath) as f:
            tree = f["EventTree"]
            trigger_n = tree["trigger_n"].array(library="np")
            root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
            hg_x = np.stack(tree["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
            hg_y = np.stack(tree["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
            xh, yh, maskh = reconstruct_hodoscope(hg_x, hg_y, threshold=4000, pitch=0.6)
            veto_wf = np.stack(tree[veto_branch].array(library="np"))
            maskv = passes_veto(veto_wf)
            if has_counters:
                one_cm_wf = np.stack(tree[one_cm_branch].array(library="np"))
                three_cm_wf = np.stack(tree[three_cm_branch].array(library="np"))
                one_cm_hit = counter_1cm_hit_mask(one_cm_wf)
                three_cm_hit = counter_3cm_hit_mask(three_cm_wf)
    except Exception as e:
        print(f"Error processing ROOT file {run}: {e}")
        return None

    # Tracker and ROOT arrays don't share an index space yet -- align_tracker_to_root_by_timestamp
    # finds spill boundaries in each stream (using real hardware timestamps), then within each
    # matched segment fits an actual clock relationship between si_data's timestamp_to_dream and
    # DREAM's FERS_Board1_tstamp_us, and matches events by nearest real time -- not by counting.
    tracker_mask, root_idx, fit_diagnostics, match_frac = align_tracker_to_root_by_timestamp(si_data, trigger_n, root_tstamp)
    n_rejected = sum(1 for _, _, fit in fit_diagnostics if fit is None)
    print(f"[{run}] Aligned {tracker_mask.sum()}/{len(si_data)} tracker events "
          f"({len(fit_diagnostics) - n_rejected}/{len(fit_diagnostics)} segments used, "
          f"match_frac={match_frac:.4%})")

    si_aligned = si_data[tracker_mask]
    xh_aligned, yh_aligned, maskh_aligned = xh[root_idx], yh[root_idx], maskh[root_idx]
    maskv_aligned = maskv[root_idx]
    one_cm_hit_aligned = one_cm_hit[root_idx] if has_counters else None
    three_cm_hit_aligned = three_cm_hit[root_idx] if has_counters else None

    # Now every array below is in the same, aligned order -- combine the
    # per-station tracker hit masks (re-sliced to the aligned subset) with
    # the hodoscope goodness mask and the veto-counter mask. The veto plane
    # fires on beam-halo particles outside the region of interest, which is
    # exactly the wide low-density background diluting the tracker/hodoscope
    # correlation, so this should cut most of that noise.
    mask = mask1[tracker_mask] & mask2[tracker_mask] & maskh_aligned
    print(f"[{run}] {mask.sum()}/{len(si_aligned)} aligned events pass all masks (incl. veto)")

    return {
        "si_aligned": si_aligned,
        "xh_aligned": xh_aligned,
        "yh_aligned": yh_aligned,
        "mask": mask,
        "maskv_aligned": maskv_aligned,
        "has_counters": has_counters,
        "one_cm_hit_aligned": one_cm_hit_aligned,
        "three_cm_hit_aligned": three_cm_hit_aligned,
    }


def process_run(run, run_output_dir):
    data = load_run_data(run)
    if data is None:
        return
    si_aligned = data["si_aligned"]
    xh_aligned, yh_aligned = data["xh_aligned"], data["yh_aligned"]
    mask = data["mask"]

    plot_list = ["", "Veto_Selection", "1_CM_Counter", "3_CM_Counter"]
    for plot in plot_list:
        if plot == "":
            mask_plot = mask
        elif plot == "Veto_Selection":
            mask_plot = mask & data["maskv_aligned"]
        elif plot == "1_CM_Counter" and data["has_counters"]:
            mask_plot = mask & data["one_cm_hit_aligned"]
        elif plot == "3_CM_Counter" and data["has_counters"]:
            mask_plot = mask & data["three_cm_hit_aligned"]
        else:
            continue
        #convert to mm
        x1, y1 = 10*si_aligned["x1"][mask_plot], 10*si_aligned["y1"][mask_plot]
        x2, y2 = 10*si_aligned["x2"][mask_plot], 10*si_aligned["y2"][mask_plot]
        xh_good, yh_good = xh_aligned[mask_plot], yh_aligned[mask_plot]

        plot_sihodocor(xh_good, yh_good, x1, y1, run, trackern="1", selection=plot, OUTPUTDIR=run_output_dir)
        plot_sihodocor(xh_good, yh_good, x2, y2, run, trackern="2", selection=plot, OUTPUTDIR=run_output_dir)

    # Sanity check that the utils.tracker calibration (fitted from exactly
    # this kind of Hodo-vs-Tracker correlation, see TRACKER_HODO_SLOPE_MEAN's
    # docstring) actually does what it claims: redo the same veto-selected
    # correlation plot as above, but on the calibrated tracker_x/y_mm
    # positions instead of raw ones -- if calibration is working, the fitted
    # slope here should come out at m ~= 1.0, unlike the uncalibrated
    # "Veto_Selection" plot above. This is a diagnostic only; it must NOT
    # feed back into get_tracker_hodo_calibration's own measurement (that
    # would be circular -- see this function's raw, uncalibrated x1/x2 above).
    x1_mm, y1_mm, x2_mm, y2_mm = calibrate_tracker_positions(
        si_aligned["x1"], si_aligned["y1"], si_aligned["x2"], si_aligned["y2"], run)
    mask_plot = mask & data["maskv_aligned"]
    xh_good, yh_good = xh_aligned[mask_plot], yh_aligned[mask_plot]
    plot_sihodocor(xh_good, yh_good, x1_mm[mask_plot], y1_mm[mask_plot], run, trackern="1",
                   selection="Veto_Selection_Calibrated", OUTPUTDIR=run_output_dir)
    plot_sihodocor(xh_good, yh_good, x2_mm[mask_plot], y2_mm[mask_plot], run, trackern="2",
                   selection="Veto_Selection_Calibrated", OUTPUTDIR=run_output_dir)


# Conversion-coefficient (fit slope) keys collected per run for --testbeam,
# one per (tracker station, axis) against the hodoscope.
SLOPE_KEYS = [("1", "x"), ("1", "y"), ("2", "x"), ("2", "y")]

# Quality-gate thresholds for trusting a run's fitted slope as a real
# conversion coefficient rather than noise -- defined in utils/tracker.py
# (see TRACKER_HODO_MIN_FIT_R's docstring there) since
# utils.tracker.get_tracker_hodo_calibration applies the exact same gate
# when reading this script's own output CSV back in as a reconstruction-time
# calibration; kept as one source of truth rather than two constants that
# could silently drift apart.
MIN_FIT_R = TRACKER_HODO_MIN_FIT_R
MIN_FIT_POINTS = TRACKER_HODO_MIN_FIT_POINTS

def get_run_slopes(run):
    """Fit Hodo-vs-Tracker{1,2}-{x,y} for one run, veto-selected, and return
    each fit's slope/r/event-count (the tracker/hodoscope conversion
    coefficient plus enough to judge whether to trust it).

    Veto-selected (rather than the unselected baseline) since that's the
    selection actually used to judge the tracker/hodoscope correlation in
    the per-run plots, and it's available for every run (unlike the 1cm/3cm
    counter selections, which only exist for runs >= 1825). Missing/failed
    fits are reported as None so a single bad run can't silently skew or
    crash the aggregate histograms.
    """
    slopes = {key: None for key in SLOPE_KEYS}
    try:
        data = load_run_data(run)
        if data is None:
            return slopes
        mask_plot = data["mask"] & data["maskv_aligned"]
        si_aligned = data["si_aligned"]
        x1, y1 = 10*si_aligned["x1"][mask_plot], 10*si_aligned["y1"][mask_plot]
        x2, y2 = 10*si_aligned["x2"][mask_plot], 10*si_aligned["y2"][mask_plot]
        xh_good, yh_good = data["xh_aligned"][mask_plot], data["yh_aligned"][mask_plot]
        n_events = int(mask_plot.sum())

        for trackern, xt, yt in [("1", x1, y1), ("2", x2, y2)]:
            fit_x = fit_profile_line(xh_good, xt)
            fit_y = fit_profile_line(yh_good, yt)
            if fit_x:
                slopes[(trackern, "x")] = {"m": fit_x["m"], "r": fit_x["r"],
                                           "n": n_events, "n_points": int(fit_x["keep"].sum())}
            if fit_y:
                slopes[(trackern, "y")] = {"m": fit_y["m"], "r": fit_y["r"],
                                           "n": n_events, "n_points": int(fit_y["keep"].sum())}
    except Exception as e:
        print(f"Error getting slopes for run {run}: {e}")
    return slopes


_SLOPE_COLORS = {
    ("1", "x"): "tab:blue",
    ("1", "y"): "tab:orange",
    ("2", "x"): "tab:green",
    ("2", "y"): "tab:red",
}

def _trusted_slopes(slopes_by_run, trackern, axis):
    """Slopes for one (tracker, axis) whose fit exists and clears both
    MIN_FIT_R and MIN_FIT_POINTS.

    A fit with too few/scattered profile points can converge on an
    essentially uncorrelated or anti-correlated line (see MIN_FIT_R's
    docstring) -- including those would silently corrupt the aggregate
    mean/std with slopes that aren't real conversion coefficients at all.
    """
    fits = [s[(trackern, axis)] for s in slopes_by_run.values() if s[(trackern, axis)] is not None]
    return np.array([f["m"] for f in fits if abs(f["r"]) >= MIN_FIT_R and f["n_points"] >= MIN_FIT_POINTS])

def plot_bigplot_slopes(slopes_by_run, testbeam):
    """Histogram the per-run conversion-coefficient (fit slope) values
    collected across a testbeam, overlaying all 4 (tracker station, axis)
    combinations on one set of axes so their distributions are directly
    comparable. Runs with no fit, or a fit below MIN_FIT_R, are excluded
    and counted rather than silently skewing the mean/std.
    """
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 9))
    testbeam_tag = testbeam.replace(" ", "_")

    trusted = {key: _trusted_slopes(slopes_by_run, *key) for key in SLOPE_KEYS}
    all_values = np.concatenate([v for v in trusted.values() if len(v)]) if any(len(v) for v in trusted.values()) else np.array([])
    if len(all_values) == 0:
        ax.text(0.5, 0.5, "No fits passed the quality cut", ha="center", va="center", transform=ax.transAxes)
    else:
        # The plotted/binned range is the 1st-99th percentile of the pooled
        # trusted slopes (plus a margin), not the raw min/max -- MIN_FIT_R
        # and MIN_FIT_POINTS catch most garbage fits, but a rare survivor
        # far from the real ~1.0 cluster would otherwise stretch a fixed
        # bin count across mostly-empty range, collapsing the actual,
        # tightly-clustered distributions into 2-3 blocky bins (exactly
        # what a hardcoded/raw-range axis did before). Percentile-based
        # bounds instead track wherever the bulk of the data actually
        # sits, so the display adapts run-to-run without manual tuning.
        lo, hi = np.percentile(all_values, [1, 99])
        margin = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1e-3) * 0.05
        lo, hi = lo - margin, hi + margin
        shared_edges = np.linspace(lo, hi, 51)
        ax.set_xlim(lo, hi)
        for key in SLOPE_KEYS:
            trackern, axis = key
            values = trusted[key]
            n_total = sum(1 for s in slopes_by_run.values() if s[key] is not None)
            n_dropped = n_total - len(values)
            if len(values) == 0:
                continue
            label = (f"Tracker{trackern} {axis.upper()} (n={len(values)}, dropped={n_dropped}, "
                     f"mean={values.mean():.4f}, std={values.std():.4f})")
            ax.hist(values, bins=shared_edges, histtype="step", lw=2,
                    color=_SLOPE_COLORS[key], label=label)
        ax.legend(fontsize=13)
    ax.set_xlabel("Fitted slope (conversion coefficient)", loc='right')
    ax.set_ylabel("Runs", loc='top')
    mh.label.exp_label(ax=ax, exp="CaloX", text=testbeam, rlabel="Hodo/Tracker Slope Distribution", data=True)
    ax.text(0.02, 0.97, f"quality cut: |r| $\\geq$ {MIN_FIT_R}, $\\geq${MIN_FIT_POINTS} profile pts",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)
    plt.tight_layout()
    outpath = os.path.join(BIGPLOT_OUTPUTDIR, f"{testbeam_tag}_slopes.png")
    plt.savefig(outpath, dpi=300)
    print("Bigplot Saved " + outpath)
    plt.close(fig)


def write_bigplot_csv(slopes_by_run, testbeam):
    """Write the per-run fitted slope/r/event-count/profile-point-count (one
    row per run, one such column group per tracker/axis) alongside the
    histogram, so individual runs -- including those the MIN_FIT_R/
    MIN_FIT_POINTS quality cut drops from the histogram -- can be inspected
    directly.
    """
    testbeam_tag = testbeam.replace(" ", "_")
    outpath = os.path.join(BIGPLOT_OUTPUTDIR, f"{testbeam_tag}_slopes.csv")
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run"]
        for t, axis in SLOPE_KEYS:
            header += [f"tracker{t}_{axis}_slope", f"tracker{t}_{axis}_r",
                       f"tracker{t}_{axis}_n", f"tracker{t}_{axis}_n_points"]
        writer.writerow(header)
        for run in sorted(slopes_by_run):
            row = slopes_by_run[run]
            values = []
            for key in SLOPE_KEYS:
                fit = row[key]
                values += [fit["m"], fit["r"], fit["n"], fit["n_points"]] if fit else ["", "", "", ""]
            writer.writerow([run] + values)
    print("Bigplot CSV Saved " + outpath)


def run_bigplot(testbeam):
    runs = get_runs_by_testbeam(testbeam)
    if not runs:
        print(f"No runs found for testbeam '{testbeam}'")
        return
    print(f"Fitting slopes for {len(runs)} runs from {testbeam} using {N_WORKERS} workers")
    slopes_by_run = {}
    with tqdm(total=len(runs), desc=testbeam, unit="run") as pbar, \
        ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(get_run_slopes, run): run for run in runs}
        for fut in as_completed(futures):
            run = futures[fut]
            pbar.set_postfix_str(f"run {run}")
            try:
                slopes_by_run[run] = fut.result()
            except Exception as e:
                tqdm.write(f"Error processing run {run}: {e}")
                slopes_by_run[run] = {key: None for key in SLOPE_KEYS}
            pbar.update(1)
    write_bigplot_csv(slopes_by_run, testbeam)
    plot_bigplot_slopes(slopes_by_run, testbeam)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct hodoscope hits from tracker data."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    parser.add_argument("--testbeam", type=str,
                         help="Instead of a single --run, fit the Hodo-vs-Tracker slope "
                              "(veto-selected) for every run in this testbeam/run period "
                              "(e.g. TB2025, TB2026, 'Cosmic 2025') and histogram the resulting "
                              "conversion coefficients, overlaid for both trackers/axes.")
    args = parser.parse_args()

    if args.testbeam:
        run_bigplot(args.testbeam)
        return

    if not args.run:
        parser.error("one of --run or --testbeam is required")

    run_output_dir = os.path.join(RUNS_OUTPUTDIR, args.run)
    os.makedirs(run_output_dir, exist_ok=True)
    process_run(args.run, run_output_dir)


if __name__ == "__main__":
    main()
