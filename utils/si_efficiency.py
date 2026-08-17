"""Hodoscope-referenced si-tracker efficiency: shared per-run loading logic.

Every event-level quantity needed by the si-tracker efficiency scripts
(``scripts/deteff_scan.py``, ``scripts/sicorr.py``, ``scripts/eff.py``) comes
from one ROOT + tracker pass per run, so it lives here once instead of being
copy-pasted per script.
"""

import numpy as np
import uproot

from .constants import HG_THRESHOLD, X_MAPPING, Y_MAPPING
from .data import get_run_filepath
from .hodo import hodo_axis_good_masks, reconstruct_hodoscope
from .selectors import get_branch_names, passes_veto
from .tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)

# Runs this short can't contain enough spill boundaries for
# align_tracker_to_root_by_timestamp to fit a clock relationship (it needs
# >= 5 spill segments) -- they're short calibration/threshold-scan blips
# tagged with a physics beam_type/energy in run_list.json, not real
# data-taking runs. Filtering them out before attempting the (expensive)
# alignment avoids both wasted work and a silent, event-count-driven bias in
# which runs happen to survive a pooled scan. Chosen well below the smallest
# legitimate run seen in practice (tens of thousands of events); real runs
# aren't close to this boundary.
MIN_EVENTS_FOR_SCAN = 500


def _read_branches(filepaths, branch_names, stack_names=()):
    """Read ``branch_names`` from ``EventTree`` across one or more ROOT files.

    ``get_run_filepath`` returns a plain string for single-file runs but a
    list of strings for runs split across multiple files -- every other
    reader in this codebase assumes the single-file case and crashes
    (``uproot.open`` rejects a list) on the rest. This accepts either and
    concatenates in the given file order, which matches run_list.json's
    order (chronological, confirmed on run 1781).

    ``stack_names`` marks branches that are per-event arrays (e.g. per-bar
    HG amplitudes) needing ``np.stack`` to become 2D before concatenation;
    everything else is treated as a flat per-event scalar branch.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    per_branch = {name: [] for name in branch_names}
    for fp in filepaths:
        with uproot.open(fp) as f:
            tree = f["EventTree"]
            for name in branch_names:
                arr = tree[name].array(library="np")
                if name in stack_names:
                    arr = np.stack(arr)
                per_branch[name].append(arr)
    return {name: np.concatenate(parts, axis=0) for name, parts in per_branch.items()}


def run_event_count(run_id):
    """Total EventTree entries for ``run_id``, summed across multi-file runs.

    Metadata-only (no branch reads), so cheap enough to use as a pre-filter
    before the much more expensive tracker load + alignment.
    """
    filepaths = get_run_filepath(run_id)
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    total = 0
    for fp in filepaths:
        with uproot.open(fp) as f:
            total += f["EventTree"].num_entries
    return total


def _read_hodo_and_timing(run_id):
    """ROOT-only per-run read: hodoscope positions/goodness, veto pass mask,
    and the timing branches the tracker aligner needs -- no tracker
    dependency at all.

    A run's tracker ``.dat`` files can be missing for reasons that have
    nothing to do with the hodoscope measurement itself (most commonly:
    tracker data for that run hasn't been synced back from the test beam
    site yet) -- there's no reason a hodoscope-vs-energy comparison should
    have to wait on that. This is split out from ``_load_si_event_data`` so
    ``load_hodo_eff_counts`` can get a real answer for every run with a
    ROOT file, independent of tracker availability/alignment.
    """
    filepaths = get_run_filepath(run_id)
    veto_branch, _, _ = get_branch_names(run_id)
    arrs = _read_branches(
        filepaths,
        ["trigger_n", "FERS_Board1_tstamp_us", "FERS_Board1_energyHG", "FERS_Board0_energyHG", veto_branch],
        stack_names=["FERS_Board1_energyHG", "FERS_Board0_energyHG", veto_branch],
    )
    trigger_n = arrs["trigger_n"]
    root_tstamp = arrs["FERS_Board1_tstamp_us"]
    xh = arrs["FERS_Board1_energyHG"][:, X_MAPPING]
    yh = arrs["FERS_Board0_energyHG"][:, Y_MAPPING]
    veto_wf = arrs[veto_branch]

    mask_v = passes_veto(veto_wf)
    xh, yh, good_hodo = reconstruct_hodoscope(xh, yh, threshold=HG_THRESHOLD)
    return trigger_n, root_tstamp, xh, yh, good_hodo, mask_v


def load_hodo_eff_counts(run_id):
    """Return ``(n_hodo_good, n_events)`` for one run.

    This is the hodoscope's own reconstruction rate over *every* event in
    the run -- unlike the si-tracker numbers below, it isn't gated by the
    veto or tracker-alignment reference selection, and (unlike
    ``load_si_and_hodo``'s hodo counts before this split) doesn't require
    the tracker to have any data for this run at all.
    """
    _, _, _, _, good_hodo, _ = _read_hodo_and_timing(run_id)
    return int(np.sum(good_hodo)), len(good_hodo)


def _tracker_hit_masks(run_id, trigger_n, root_tstamp, per_axis=False, x_root=None, good_root=None):
    """Return ``(root_mask1, root_mask2)``: per-ROOT-event booleans for
    whether si tracker station 1 / 2 has a trustworthy hit, aligned onto
    ROOT's own event ordering. With ``per_axis=True``, also returns the four
    single-axis masks those are built from: ``(root_mask1, root_mask2,
    x1_mask, y1_mask, x2_mask, y2_mask)``.

    ``x_root``/``good_root`` (typically the hodoscope's own x/good_hodo)
    are passed straight through to ``align_tracker_to_root_by_timestamp``'s
    real-correlation QA check -- see ``MIN_ALIGNMENT_CORRELATION`` for why
    this matters: match_frac alone lets through alignments that satisfy
    their own self-consistency checks without finding true per-event
    correspondences (confirmed directly: runs 1859-1866 e+ 40 GeV had
    match_frac 0.69-0.84 but real correlation |r| < 0.02). Omit to skip
    that check.

    Raises if the tracker has no ``.dat`` data for this run at all, or if
    ``build_aligned_tracker_branches``'s own alignment-quality gates reject
    it (too few spill boundaries, inconsistent boundary matching, too low a
    match fraction, or -- when x_root/good_root are given -- too weak a
    real correlation; see that function's docstring).
    """
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
    tracker_branches, match_frac = build_aligned_tracker_branches(
        si_data, trigger_n, root_tstamp, x_root=x_root, good_root=good_root)
    x1, y1 = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
    x2, y2 = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]

    # Not every ROOT event with a good hodoscope+veto hit has a matching
    # tracker row -- the tracker silently drops a sizeable, run-dependent
    # fraction of triggers (row just isn't written). Those are genuine
    # tracker misses, not missing data, so the reference frame here is
    # every ROOT event (not only the ones that matched); a sentinel value
    # means "miss" (either genuinely no tracker row, or no trustworthy
    # match), same as a station's own no-hit sentinel would.
    x1_mask, y1_mask = x1 != SENTINEL_X1, y1 != SENTINEL_Y1
    x2_mask, y2_mask = x2 != SENTINEL_X2, y2 != SENTINEL_Y2
    root_mask1 = x1_mask & y1_mask
    root_mask2 = x2_mask & y2_mask
    if per_axis:
        return root_mask1, root_mask2, x1_mask, y1_mask, x2_mask, y2_mask
    return root_mask1, root_mask2


def _load_si_event_data(run_id):
    """Shared internals for one run: hodoscope positions, the reference
    selection mask, and the per-station hit masks -- all indexed over every
    ROOT event (not just the ones the tracker wrote a row for).

    Raises on any failure (missing ROOT/tracker file, bad tracker-to-ROOT
    alignment, etc.) so callers can decide how to handle it per run.
    """
    trigger_n, root_tstamp, xh, yh, good_hodo, mask_v = _read_hodo_and_timing(run_id)
    root_mask1, root_mask2 = _tracker_hit_masks(run_id, trigger_n, root_tstamp,
                                                x_root=xh, good_root=good_hodo)
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


def load_si_and_hodo(run_id):
    """Combine the si-tracker and hodoscope numbers into a single ROOT-file
    pass -- for callers (e.g. deteff_scan.py) that want both per run without
    paying the I/O cost twice, and without losing a run's hodoscope numbers
    just because its tracker data isn't available.

    Returns ``(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2,
    n_hodo_good, n_events, tracker_error)``. ``tracker_error`` is ``None``
    when the tracker aligned successfully; otherwise it's the reason it
    couldn't be used (see ``_tracker_hit_masks``) and
    xh_ref/xh_sel1/yh_sel1/xh_sel2/yh_sel2 come back empty -- but
    n_hodo_good/n_events are always valid, since they come straight from
    ROOT with no tracker dependency at all.
    """
    trigger_n, root_tstamp, xh, yh, good_hodo, mask_v = _read_hodo_and_timing(run_id)
    n_hodo_good, n_events = int(np.sum(good_hodo)), len(good_hodo)

    empty = np.array([])
    try:
        root_mask1, root_mask2 = _tracker_hit_masks(run_id, trigger_n, root_tstamp,
                                                     x_root=xh, good_root=good_hodo)
        ref = good_hodo & mask_v
        xh_ref, yh_ref = xh[ref], yh[ref]
        xh_sel1, yh_sel1 = xh[ref & root_mask1], yh[ref & root_mask1]
        xh_sel2, yh_sel2 = xh[ref & root_mask2], yh[ref & root_mask2]
        tracker_error = None
    except Exception as e:
        xh_ref = yh_ref = xh_sel1 = yh_sel1 = xh_sel2 = yh_sel2 = empty
        tracker_error = str(e)

    return xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good, n_events, tracker_error


def load_axis_report(run_id):
    """Per-axis/per-plane/track hit masks for one run, over every ROOT event.

    Unlike the reference-gated functions above (``load_si_ref_and_hits`` and
    friends, which condition on hodoscope-good + veto-pass events, matching
    what deteff_scan.py/sicorr.py need), this reports each axis and plane's
    raw hit rate over *all* events -- for ``scripts/eff.py``'s per-run
    report (each axis, each plane, and the tracker/tracker+hodo coincidence).
    No veto gating; raises if the tracker has no usable data for this run.

    Returns a dict with ``n_events`` and, for each of ``x1``, ``y1``, ``x2``,
    ``y2``, ``hodo_x``, ``hodo_y``, a boolean mask (length ``n_events``)
    marking whether that axis/plane registered a hit -- callers combine them
    as needed (e.g. ``x1 & y1`` for station 1's 2D hit rate).
    """
    filepaths = get_run_filepath(run_id)
    arrs = _read_branches(
        filepaths,
        ["trigger_n", "FERS_Board1_tstamp_us", "FERS_Board1_energyHG", "FERS_Board0_energyHG"],
        stack_names=["FERS_Board1_energyHG", "FERS_Board0_energyHG"],
    )
    trigger_n = arrs["trigger_n"]
    root_tstamp = arrs["FERS_Board1_tstamp_us"]
    hg_x = arrs["FERS_Board1_energyHG"][:, X_MAPPING]
    hg_y = arrs["FERS_Board0_energyHG"][:, Y_MAPPING]

    hodo_x, hodo_y = hodo_axis_good_masks(hg_x, hg_y, threshold=HG_THRESHOLD)
    xh, _, _ = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD)
    root_mask1, root_mask2, x1_mask, y1_mask, x2_mask, y2_mask = _tracker_hit_masks(
        run_id, trigger_n, root_tstamp, per_axis=True, x_root=xh, good_root=hodo_x & hodo_y)

    return {
        "n_events": len(hg_x),
        "x1": x1_mask, "y1": y1_mask, "x2": x2_mask, "y2": y2_mask,
        "hodo_x": hodo_x, "hodo_y": hodo_y,
    }
