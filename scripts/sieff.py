import os
import argparse
import numpy as np
import uproot

from utils.data import get_run_filepath
from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)
from utils.plotting import get_beam_label, plot_effhist2d, intrinsic_efficiency
from utils.selectors import passes_veto, get_branch_names
from utils.hodo import reconstruct_hodoscope
from utils.constants import HG_THRESHOLD, PITCH, VETO_THRESHOLD, X_MAPPING, Y_MAPPING

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/sieff"
os.makedirs(OUTPUTDIR, exist_ok=True)


def _load_si_event_data(run_id):
    """Shared internals for one run: hodoscope positions, the reference
    selection mask, and the per-station hit masks -- all indexed over every
    ROOT event (not just the ones the tracker wrote a row for).

    Raises on any failure (missing ROOT/tracker file, bad tracker-to-ROOT
    alignment, etc.) so callers can decide how to handle it per run.
    """
    filepath = get_run_filepath(run_id)
    veto, _, _ = get_branch_names(run_id)
    with uproot.open(filepath) as f:
        tree = f["EventTree"]
        trigger_n = tree["trigger_n"].array(library="np")
        root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
        xh = np.stack(tree["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
        yh = np.stack(tree["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
        veto_wf = np.stack(tree[veto].array(library="np"))

    mask_v = passes_veto(veto_wf)
    xh, yh, good_hodo = reconstruct_hodoscope(xh, yh)
    si_data = load_tracker_run(run_id)

    # build_aligned_tracker_branches matches tracker events to ROOT by real
    # clock time (see align_tracker_to_root_by_timestamp's docstring), and
    # re-indexes tracker x1/y1/x2/y2 onto ROOT's own event ordering, filling
    # unmatched events with the tracker's existing no-hit sentinels. An
    # earlier counting-based version of this alignment silently mismatched
    # a run-dependent fraction of events whenever the true tracker<->ROOT
    # offset drifted within a run; matching by time instead measurably
    # improved per-event correctness (confirmed: raw position correlation
    # against the hodoscope up from 0.66-0.81 to 0.74-0.82 across test runs).
    tracker_branches, match_frac = build_aligned_tracker_branches(si_data, trigger_n, root_tstamp)
    x1, y1 = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
    x2, y2 = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]

    # Not every ROOT event with a good hodoscope+veto hit has a matching
    # tracker row -- the tracker silently drops a sizeable, run-dependent
    # fraction of triggers (row just isn't written). Those are genuine
    # tracker misses, not missing data, so the reference frame here is
    # every ROOT event (not only the ones that matched); a sentinel value
    # means "miss" (either genuinely no tracker row, or no trustworthy
    # match), same as a station's own no-hit sentinel would.
    root_mask1 = (x1 != SENTINEL_X1) & (y1 != SENTINEL_Y1)
    root_mask2 = (x2 != SENTINEL_X2) & (y2 != SENTINEL_Y2)

    ref = good_hodo & mask_v
    return xh, yh, good_hodo, ref, root_mask1, root_mask2


def load_si_ref_and_hits(run_id):
    """Return ``(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2)`` for one run.

    ``xh_ref``/``yh_ref`` are hodoscope-reconstructed positions for every
    event passing the hodoscope-goodness + veto reference selection;
    ``xh_sel1``/``xh_sel2`` are the subset of those that also registered a
    hit on si tracker station 1 / 2, respectively.
    """
    xh, yh, good_hodo, ref, root_mask1, root_mask2 = _load_si_event_data(run_id)
    xh_ref, yh_ref = xh[ref], yh[ref]
    xh_sel1, yh_sel1 = xh[ref & root_mask1], yh[ref & root_mask1]
    xh_sel2, yh_sel2 = xh[ref & root_mask2], yh[ref & root_mask2]
    return xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2


def load_si_hit_masks(run_id):
    """Return ``(hit1, hit2)``: per-station hit booleans for every reference-selected event.

    Same event order/length for both, so ``hit1 & ~hit2`` etc. are directly
    comparable -- meant for checking whether the two stations' misses are
    correlated (e.g. a shared upstream cause) rather than independent.
    """
    xh, yh, good_hodo, ref, root_mask1, root_mask2 = _load_si_event_data(run_id)
    return root_mask1[ref], root_mask2[ref]


def load_hodo_eff_counts(run_id):
    """Return ``(n_hodo_good, n_events)`` for one run.

    This is the hodoscope's own reconstruction rate over *every* event in
    the run -- unlike the si-tracker numbers above, it isn't gated by the
    veto or tracker-alignment reference selection.
    """
    xh, yh, good_hodo, ref, root_mask1, root_mask2 = _load_si_event_data(run_id)
    return int(np.sum(good_hodo)), len(good_hodo)


def load_si_and_hodo(run_id):
    """Combine ``load_si_ref_and_hits`` and ``load_hodo_eff_counts`` into a
    single ROOT-file pass -- for callers (e.g. sieff_scan.py) that need both
    per run and don't want to pay the I/O cost twice.

    Returns ``(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good, n_events)``.
    """
    xh, yh, good_hodo, ref, root_mask1, root_mask2 = _load_si_event_data(run_id)
    xh_ref, yh_ref = xh[ref], yh[ref]
    xh_sel1, yh_sel1 = xh[ref & root_mask1], yh[ref & root_mask1]
    xh_sel2, yh_sel2 = xh[ref & root_mask2], yh[ref & root_mask2]
    n_hodo_good, n_events = int(np.sum(good_hodo)), len(good_hodo)
    return xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good, n_events


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()

    try:
        xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2 = load_si_ref_and_hits(args.run)
    except Exception as e:
        print(f"Error processing run {args.run}: {e}")
        return

    runtype = get_beam_label(args.run)
    sub_outdir = os.path.join(OUTPUTDIR, args.run)
    os.makedirs(sub_outdir, exist_ok=True)

    eff1, h_ref1, *_ = plot_effhist2d(xh_ref, yh_ref, xh_sel1, yh_sel1, 64, "Hodo X [mm]", "Hodo Y [mm]",
                   "Si Tracker 1 Efficiency", os.path.join(sub_outdir, f"si1_effmap_{args.run}.png"),
                   runtype=runtype)
    eff2, h_ref2, *_ = plot_effhist2d(xh_ref, yh_ref, xh_sel2, yh_sel2, 64, "Hodo X [mm]", "Hodo Y [mm]",
                   "Si Tracker 2 Efficiency", os.path.join(sub_outdir, f"si2_effmap_{args.run}.png"),
                   runtype=runtype)

    eff1_mean, eff1_unc = intrinsic_efficiency(eff1, h_ref1)
    eff2_mean, eff2_unc = intrinsic_efficiency(eff2, h_ref2)
    print(f"Run {args.run}: n_ref={len(xh_ref)}  "
          f"intrinsic_eff_si1={eff1_mean:.3f} ± {(eff1_unc or 0):.3f}  "
          f"intrinsic_eff_si2={eff2_mean:.3f} ± {(eff2_unc or 0):.3f}")


if __name__ == "__main__":
    main()
