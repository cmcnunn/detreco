"""Hodoscope-referenced 2D efficiency maps for MCP1, MCP2, and the Veto."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import HG_THRESHOLD, X_MAPPING, Y_MAPPING, PITCH, VETO_THRESHOLD
from utils.data import get_run_filepath
from utils.hodo import reconstruct_hodoscope
from utils.io import ensure_output_dir
from utils.plotting import get_beam_label, intrinsic_efficiency, plot_effhist2d
from utils.selectors import (
    COUNTER_1CM_3CM_FIRST_RUN,
    counter_1cm_hit_mask,
    counter_3cm_hit_mask,
    get_branch_names,
    get_counter_branch_names,
    get_mcp_pulse_window_ns,
    passes_veto,
)
from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)
from utils.waveforms import subtract_baseline

# --- Branch names ---
Y_HG = "FERS_Board0_energyHG"
X_HG = "FERS_Board1_energyHG"

# --- Thresholds ---
LG_THRESHOLD = 229
DRS_THRESHOLD = -13.6

# --- MCP pulse finding (matches mcp_studies.py) ---
# Pulse window (ns) is run-dependent; see utils.selectors.get_mcp_pulse_window_ns.
MCP_NOISE_SAMPLES = 50
MCP_NOISE_MARGIN_ADC = 20
MCP_MIN_PULSE_FWHM_NS = 1.0

# --- Geometry ---
SAMPLE_NS = 0.2

OUTPUT_DIR = ensure_output_dir("effplots")



def _mcp_hit_mask(mcp_bs, pulse_window_ns):
    """True for events with a pulse minimum below dynamic threshold inside the window."""
    w_start = int(pulse_window_ns[0] / SAMPLE_NS)
    w_end = int(pulse_window_ns[1] / SAMPLE_NS)
    noise_avg = np.mean(np.abs(mcp_bs[:, :MCP_NOISE_SAMPLES]), axis=1)
    thresh = -(noise_avg + MCP_NOISE_MARGIN_ADC)
    return mcp_bs[:, w_start:w_end + 1].min(axis=1) < thresh

def get_intrinsic_efficiency(x_ref, y_ref, x_sel, y_sel, return_uncertainty=True):
    """Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."""
    h_ref, xedges, yedges = np.histogram2d(x_ref, y_ref, bins=64)
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[xedges, yedges])

    eff = np.divide(h_sel, h_ref, out=np.zeros_like(h_sel, dtype=float), where=h_ref > 0)
    geometric_mask = eff > 0.5
    eff = eff[geometric_mask] #mask to only consider bins where there are reference hits
    intrinsic_efficiency = np.mean(eff)  
    if return_uncertainty:
        # Calculate uncertainty using binomial statistics
        n_bins = np.sum(geometric_mask)
        if n_bins > 0:
            uncertainty = np.sqrt(intrinsic_efficiency * (1 - intrinsic_efficiency) / n_bins)
            return intrinsic_efficiency, uncertainty
        else:
            return intrinsic_efficiency, None
    else:
        return intrinsic_efficiency

_EMPTY = np.array([])


def process_single_run(run_data):
    """Returns a dict of per-detector ``(x_ref, y_ref, x_sel, y_sel)`` tuples.

    ``mcp1``/``mcp2``/``veto`` use every run's hodoscope-good events as their
    reference (available for all runs). ``1cm_counter``/``3cm_counter`` use
    their own, separately-tracked reference -- restricted to runs >=
    ``COUNTER_1CM_3CM_FIRST_RUN`` -- since those branches don't exist in
    earlier runs; mixing in earlier runs' hodo-good events as "reference"
    would inflate the denominator with events that can never register a
    counter hit, artificially deflating the efficiency.
    """
    run_id, file_path = run_data
    VETO, MCP1, MCP2 = get_branch_names(run_id)
    pulse_window_ns = get_mcp_pulse_window_ns(run_id)
    has_counters = int(run_id) >= COUNTER_1CM_3CM_FIRST_RUN
    if has_counters:
        ONE_CM, THREE_CM = get_counter_branch_names(run_id)
    empty = {"mcp1": (_EMPTY, _EMPTY, _EMPTY, _EMPTY), "mcp2": (_EMPTY, _EMPTY, _EMPTY, _EMPTY),
             "veto": (_EMPTY, _EMPTY, _EMPTY, _EMPTY), "1cm_counter": (_EMPTY, _EMPTY, _EMPTY, _EMPTY),
             "3cm_counter": (_EMPTY, _EMPTY, _EMPTY, _EMPTY)}
    try:
        with uproot.open(file_path) as f:
            tree = f["EventTree"]
            hg_x = np.stack(tree[X_HG].array(library="np"))[:, X_MAPPING]
            hg_y = np.stack(tree[Y_HG].array(library="np"))[:, Y_MAPPING]
            veto = np.stack(tree[VETO].array(library="np"))
            mcp1 = subtract_baseline(np.stack(tree[MCP1].array(library="np")))
            mcp2 = subtract_baseline(np.stack(tree[MCP2].array(library="np")))
            if has_counters:
                one_cm_wf = np.stack(tree[ONE_CM].array(library="np"))
                three_cm_wf = np.stack(tree[THREE_CM].array(library="np"))

        xh, yh, good_hodo = reconstruct_hodoscope(
            hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH, method="mean",
        )

        veto_sel = passes_veto(veto, threshold=VETO_THRESHOLD)
        mcp1_hit = _mcp_hit_mask(mcp1, pulse_window_ns)
        mcp2_hit = _mcp_hit_mask(mcp2, pulse_window_ns)
        if has_counters:
            one_cm_hit = counter_1cm_hit_mask(one_cm_wf)
            three_cm_hit = counter_3cm_hit_mask(three_cm_wf)

        w_start = int(pulse_window_ns[0] / SAMPLE_NS)
        w_end = int(pulse_window_ns[1] / SAMPLE_NS)
        noise_avg = np.mean(np.abs(mcp1[:, :MCP_NOISE_SAMPLES]), axis=1)
        ref = good_hodo
        intrinsic_efficiency_mcp1, uncertainty_mcp1 = get_intrinsic_efficiency(xh[ref], yh[ref],
                                                                               xh[ref & mcp1_hit], yh[ref & mcp1_hit])
        intrinsic_efficiency_mcp2, uncertainty_mcp2 = get_intrinsic_efficiency(xh[ref], yh[ref],
                                                                               xh[ref & mcp2_hit], yh[ref & mcp2_hit])
        intrinsic_efficiency_veto, uncertainty_veto = get_intrinsic_efficiency(xh[ref], yh[ref],
                                                                               xh[ref & veto_sel], yh[ref & veto_sel])
        log = (f"  [{run_id}] n_events={len(mcp1)}  "
              f"mcp1_hits={mcp1_hit.sum()}  mcp2_hits={mcp2_hit.sum()}  veto_hits={veto_sel.sum()}  "
              f"mean_thresh={-(noise_avg.mean() + MCP_NOISE_MARGIN_ADC):.1f}  "
              f"mean_window_min={mcp1[:, w_start:w_end+1].min(axis=1).mean():.1f} "
              f"intrinsic_eff_mcp1={intrinsic_efficiency_mcp1:.3f} ± {uncertainty_mcp1:.3f}  "
              f"intrinsic_eff_mcp2={intrinsic_efficiency_mcp2:.3f} ± {uncertainty_mcp2:.3f}  "
              f"intrinsic_eff_veto={intrinsic_efficiency_veto:.3f} ± {uncertainty_veto:.3f}")

        result = {
            "mcp1": (xh[ref], yh[ref], xh[ref & mcp1_hit], yh[ref & mcp1_hit]),
            "mcp2": (xh[ref], yh[ref], xh[ref & mcp2_hit], yh[ref & mcp2_hit]),
            "veto": (xh[ref], yh[ref], xh[ref & veto_sel], yh[ref & veto_sel]),
            "1cm_counter": empty["1cm_counter"],
            "3cm_counter": empty["3cm_counter"],
        }
        if has_counters:
            intrinsic_efficiency_1cm, uncertainty_1cm = get_intrinsic_efficiency(
                xh[ref], yh[ref], xh[ref & one_cm_hit], yh[ref & one_cm_hit])
            intrinsic_efficiency_3cm, uncertainty_3cm = get_intrinsic_efficiency(
                xh[ref], yh[ref], xh[ref & three_cm_hit], yh[ref & three_cm_hit])
            log += (f"  1cm_hits={one_cm_hit.sum()}  3cm_hits={three_cm_hit.sum()}  "
                    f"intrinsic_eff_1cm={intrinsic_efficiency_1cm:.3f} ± {(uncertainty_1cm or 0):.3f}  "
                    f"intrinsic_eff_3cm={intrinsic_efficiency_3cm:.3f} ± {(uncertainty_3cm or 0):.3f}")
            result["1cm_counter"] = (xh[ref], yh[ref], xh[ref & one_cm_hit], yh[ref & one_cm_hit])
            result["3cm_counter"] = (xh[ref], yh[ref], xh[ref & three_cm_hit], yh[ref & three_cm_hit])
        print(log)
        return result
    except Exception as e:
        print(f"Error in {run_id}: {e}")
        return empty


def process_tracker_referenced_run(run_id, output_dir):
    """Silicon-tracker-referenced efficiency maps for one run.

    Mirrors ``process_single_run``/``plot_effhist2d`` above but with the
    reference/selected roles flipped: the reference position comes from
    whichever tracker station registered a hit (its own (x, y), not the
    hodoscope's), and the "selected" detectors are the hodoscope, veto, MCP1,
    MCP2, and (where available, run >= ``COUNTER_1CM_3CM_FIRST_RUN``) the
    1cm/3cm test counters. Produces one set of maps per station
    (station1_hit_mask / station2_hit_mask), since either station can serve
    as the reference.
    """
    filepath = get_run_filepath(run_id)
    veto_branch, mcp1_branch, mcp2_branch = get_branch_names(run_id)
    pulse_window_ns = get_mcp_pulse_window_ns(run_id)
    has_counters = int(run_id) >= COUNTER_1CM_3CM_FIRST_RUN
    if has_counters:
        one_cm_branch, three_cm_branch = get_counter_branch_names(run_id)

    with uproot.open(filepath) as f:
        tree = f["EventTree"]
        trigger_n = tree["trigger_n"].array(library="np")
        root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
        hg_x = np.stack(tree[X_HG].array(library="np"))[:, X_MAPPING]
        hg_y = np.stack(tree[Y_HG].array(library="np"))[:, Y_MAPPING]
        veto_wf = np.stack(tree[veto_branch].array(library="np"))
        mcp1_wf = subtract_baseline(np.stack(tree[mcp1_branch].array(library="np")))
        mcp2_wf = subtract_baseline(np.stack(tree[mcp2_branch].array(library="np")))
        if has_counters:
            one_cm_wf = np.stack(tree[one_cm_branch].array(library="np"))
            three_cm_wf = np.stack(tree[three_cm_branch].array(library="np"))

    xh, yh, good_hodo = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH)
    veto_sel = passes_veto(veto_wf, threshold=VETO_THRESHOLD)
    mcp1_hit = _mcp_hit_mask(mcp1_wf, pulse_window_ns)
    mcp2_hit = _mcp_hit_mask(mcp2_wf, pulse_window_ns)
    if has_counters:
        one_cm_hit = counter_1cm_hit_mask(one_cm_wf)
        three_cm_hit = counter_3cm_hit_mask(three_cm_wf)

    si_data = load_tracker_run(run_id)
    tracker_branches, match_frac = build_aligned_tracker_branches(si_data, trigger_n, root_tstamp)
    x1, y1 = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
    x2, y2 = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]
    matched = tracker_branches["tracker_matched"]
    print(f"  [{run_id}] Aligned {matched.sum()}/{len(trigger_n)} ROOT events "
          f"(tracker-side match_frac={match_frac:.4%})")

    runtype = get_beam_label(run_id)
    # Both quantities live in ROOT-event space now (build_aligned_tracker_branches
    # already re-indexed the tracker onto it), so no more root_idx bookkeeping --
    # a plain sentinel check replaces station1_hit_mask/station2_hit_mask +
    # tracker_mask, and every other array (good_hodo, veto_sel, ...) is already
    # ROOT-indexed and needs no re-indexing at all.
    station_data = {
        1: (10 * x1, 10 * y1, (x1 != SENTINEL_X1) & (y1 != SENTINEL_Y1)),
        2: (10 * x2, 10 * y2, (x2 != SENTINEL_X2) & (y2 != SENTINEL_Y2)),
    }
    for n, (x_tr, y_tr, ref) in station_data.items():
        x_ref, y_ref = x_tr[ref], y_tr[ref]
        if len(x_ref) == 0:
            print(f"  [{run_id}] Tracker{n}: no reference hits, skipping")
            continue
        x_range, y_range = (x_ref.min(), x_ref.max()), (y_ref.min(), y_ref.max())

        selections = [("hodoscope", good_hodo), ("veto", veto_sel),
                      ("mcp1", mcp1_hit), ("mcp2", mcp2_hit)]
        if has_counters:
            selections += [("1cm_counter", one_cm_hit), ("3cm_counter", three_cm_hit)]

        for name, sel_mask in selections:
            sel = ref & sel_mask
            x_sel, y_sel = x_tr[sel], y_tr[sel]
            filename = os.path.join(output_dir, f"tracker{n}_{name}effmap_{run_id}.png")
            eff, h_ref, *_ = plot_effhist2d(
                x_ref, y_ref, x_sel, y_sel, 120,
                "Silicon Tracker X [mm]", "Silicon Tracker Y [mm]",
                f"{name.replace('_', ' ').title()} vs Tracker{n}",
                filename, runtype=runtype, x_range=x_range, y_range=y_range,
            )
            mean, unc = intrinsic_efficiency(eff, h_ref)
            print(f"  [{run_id}] Tracker{n} ref -> {name}: n_ref={len(x_ref)} "
                  f"intrinsic_eff={mean:.3f} ± {(unc or 0):.3f}")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run ID to process (default: all runs)")
    parser.add_argument("--reference", choices=["hodo", "tracker"], default="hodo",
                        help="Reference detector for the efficiency maps: 'hodo' (default, "
                             "checks MCP1/MCP2/veto against the hodoscope, aggregating all "
                             "runs unless --run is given) or 'tracker' (checks hodoscope/veto/"
                             "1cm/3cm counters against each si-tracker station; requires --run)")
    args = parser.parse_args()

    if args.reference == "tracker":
        if args.run is None:
            raise SystemExit("--reference tracker requires --run (per-run alignment, not aggregated)")
        process_tracker_referenced_run(args.run, OUTPUT_DIR)
        return

    try:
        with open("run_list.json", "r") as f:
            run_files = json.load(f)
    except Exception:
        with open("data/run_list.json", "r") as f:
            run_files = json.load(f)

    if args.run is not None:
        if args.run not in run_files:
            raise SystemExit(f"Run '{args.run}' not found in run_list.json")
        run_files = {args.run: run_files[args.run]}

    runs = [(run_id, entry["file"]) for run_id, entry in run_files.items()]
    runs_label = ", ".join(r[0] for r in runs)

    print(f"Processing {len(runs)} run(s) sequentially...")
    results = [process_single_run(run) for run in runs]

    plt.style.use(mh.style.ROOT)
    detectors = [("mcp1", "MCP1"), ("mcp2", "MCP2"), ("veto", "Veto"),
                 ("1cm_counter", "1cm Counter"), ("3cm_counter", "3cm Counter")]

    for key, label in detectors:
        x_ref = np.concatenate([r[key][0] for r in results])
        y_ref = np.concatenate([r[key][1] for r in results])
        x_sel = np.concatenate([r[key][2] for r in results])
        y_sel = np.concatenate([r[key][3] for r in results])
        if len(x_ref) == 0:
            print(f"No reference events for {label} (e.g. no runs >= "
                  f"{COUNTER_1CM_3CM_FIRST_RUN} in this selection); skipping")
            continue
        filename = os.path.join(OUTPUT_DIR, f"hodo_{key}effmap_{runs_label}.png")
        runtype = get_beam_label(args.run)
        # Deliberately *not* passing x_range/y_range here: x_ref/y_ref are
        # hodoscope positions quantized to PITCH/2 (reconstruct_hodoscope's
        # "mean" method). Binning them into 64 bins over the empirical
        # (x_ref.min(), x_ref.max()) box makes the bin width a few percent
        # off from that quantization step, so the bin phase drifts across
        # the range and one bin ends up landing almost exactly between two
        # quantized values -- an aliasing artifact that looks like a dead
        # bar (confirmed: neighboring bins ~900-1000 events, the aliased
        # bin ~140, on run 1832). Leaving these unset falls back to
        # compute_efficiency_map's default +/-32*PITCH grid, which is
        # exactly bar-pitch-aligned and has no such drift.
        eff, h_ref, *_ = plot_effhist2d(
                x_ref, y_ref, x_sel, y_sel, 64,
                "Silicon Tracker X [mm]", "Silicon Tracker Y [mm]",
                f"{key.replace('_', ' ').title()} vs Hodoscope",
                filename, runtype=runtype
        )
        print(f"Efficiency map for {label} saved to {filename}")

    print("Aggregation complete. Generating plots...")
    print("Done.")


if __name__ == "__main__":
    main()
