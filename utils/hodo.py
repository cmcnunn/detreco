"""Hodoscope (HG fiber plane) reconstruction.

Every script that uses the hodoscope had its own copy of this logic. It's the
same recipe every time: threshold the per-bar HG amplitudes, check that the
pattern of hits is consistent with a single particle passing through (one or
two *adjacent* bars fired), and convert the best bar index into a transverse
position in mm. Given a run ID, the positions are then rotated into the
silicon tracker's frame (``rotate_to_tracker_frame``).
"""

import numpy as np

from .constants import HG_THRESHOLD, PITCH
from .data import get_run_hodo_rotation

# Hodoscope has 64 bars per plane. Bar index 0 corresponds to one edge,
# bar index 63 to the other; the geometric center sits between bar 31 and
# bar 32, i.e. at half-index 31.5. We multiply by PITCH to get mm.
N_BARS = 64
BAR_CENTER_OFFSET = (N_BARS - 1) / 2.0  # 31.5


def calculate_bar_position(bar_index, pitch=PITCH):
    """Convert a bar index (0..63) to a transverse position in mm.

    Position is zero at the geometric center of the hodoscope.
    """
    return (np.asarray(bar_index) - BAR_CENTER_OFFSET) * pitch


def rotate_to_tracker_frame(x, y, dx_dy, dy_dx):
    """Undo the hodoscope's rotation relative to the silicon tracker.

    ``scripts.hodores`` measures, per run, how the hodoscope sees a
    two-station track extrapolated to its z:

        hodo_x - x_track = dx_dy * y_track + c_x
        hodo_y - y_track = dy_dx * x_track + c_y

    i.e. hodo = M @ track + c with M = [[1, dx_dy], [dy_dx, 1]] -- a ~27-31
    mrad rotation, (dx_dy - dy_dx) / 2, plus a ~5 mrad non-orthogonality
    between the X and Y bar planes, dx_dy + dy_dx. This applies M^-1 about
    the hodoscope's own centre. The offsets c are not applied: the hodoscope
    keeps its own origin, which was never surveyed to the tracker's.

    Mixes the two planes, so an event that fails the goodness cut on one
    plane carries that plane's meaningless position into the other -- mask
    with ``good_hodo_mask`` as before, not one plane's
    ``hodo_axis_good_masks``.
    """
    det = 1.0 - dx_dy * dy_dx
    return (x - dx_dy * y) / det, (y - dy_dx * x) / det


def hodo_axis_good_masks(hg_x, hg_y, threshold=HG_THRESHOLD):
    """Per-axis boolean masks: does each plane, independently, have a
    well-formed hit pattern (>=1 bar above ``threshold``, and those bars
    forming a contiguous group, i.e. span == n_hits - 1)?

    Split out from ``good_hodo_mask`` so callers that want each plane's own
    rate (e.g. a per-run X/Y efficiency report) don't have to duplicate this
    logic -- ``good_hodo_mask`` is just the AND of the two.
    """
    hit_x = hg_x > threshold
    hit_y = hg_y > threshold
    n_hit_x = hit_x.sum(axis=1)
    n_hit_y = hit_y.sum(axis=1)

    idx = np.arange(N_BARS)
    with np.errstate(all="ignore"):
        x_idx = np.where(hit_x, idx, np.nan)
        y_idx = np.where(hit_y, idx, np.nan)
        span_x = np.nanmax(x_idx, axis=1) - np.nanmin(x_idx, axis=1)
        span_y = np.nanmax(y_idx, axis=1) - np.nanmin(y_idx, axis=1)

        # Contiguity requirement (no gaps): span == n_hits - 1
        contiguous_x = span_x == (n_hit_x - 1)
        contiguous_y = span_y == (n_hit_y - 1)

    good_x = (n_hit_x >= 1) & contiguous_x
    good_y = (n_hit_y >= 1) & contiguous_y
    return good_x, good_y


def good_hodo_mask(hg_x, hg_y, threshold=HG_THRESHOLD):
    """Boolean mask of events with a well-formed hodoscope hit pattern.

    An event is "good" when each plane has between ``min_hits`` and
    ``max_hits`` bars above ``threshold``, and those bars form a contiguous
    group (span == n_hits - 1 <= ``max_span``).
    """
    good_x, good_y = hodo_axis_good_masks(hg_x, hg_y, threshold=threshold)
    return good_x & good_y


def reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH,
                          method="argmax", run_id=None):
    """Reconstruct (x, y) per event and return with a goodness mask.

    Parameters
    ----------
    hg_x, hg_y : array_like, shape (n_events, 64)
        Per-bar HG amplitudes for the X and Y planes.
    threshold : float, optional
        HG amplitude above which a bar is considered "hit".
    pitch : float, optional
        Bar pitch in mm.
    method : {"argmax", "mean"}, optional
        ``"argmax"`` picks the single bar with the highest HG per plane (robust
        when two adjacent bars fired). ``"mean"`` returns the mean index of
        all hit bars (equivalent for 1-hit events; halfway between centers for
        2-hit events).
    run_id : optional
        If given, rotate the positions into the silicon tracker's frame with
        this run's ``hodo_rotation`` from run_list.json (see
        ``rotate_to_tracker_frame``). They then no longer sit exactly on the
        bar grid. ``None`` (the default) returns the unrotated bar frame --
        what ``scripts.hodores`` needs to measure the rotation in the first
        place.

    Returns
    -------
    x_rec, y_rec : ndarray, shape (n_events,)
        Reconstructed hit positions in mm. Values for events that fail the
        goodness cut are still computed but should be masked off by the caller.
    good_hodo : ndarray of bool, shape (n_events,)
        True where the hit pattern is well-formed (see ``good_hodo_mask``).
    """
    hg_x = np.asarray(hg_x)
    hg_y = np.asarray(hg_y)

    hit_x = hg_x > threshold
    hit_y = hg_y > threshold
    good_hodo = good_hodo_mask(hg_x, hg_y, threshold=threshold)

    if method == "argmax":
        # Pick the strongest-amplitude bar per plane. `np.where(hit, hg, 0)`
        # ensures non-hit bars can never win the argmax.
        x_idx = np.argmax(np.where(hit_x, hg_x, 0), axis=1)
        y_idx = np.argmax(np.where(hit_y, hg_y, 0), axis=1)
        x_rec = calculate_bar_position(x_idx, pitch)
        y_rec = calculate_bar_position(y_idx, pitch)
    elif method == "mean":
        idx = np.arange(N_BARS)
        x_idx_map = np.where(hit_x, idx, np.nan)
        y_idx_map = np.where(hit_y, idx, np.nan)
        with np.errstate(all="ignore"):
            mean_x = np.nanmean(x_idx_map, axis=1)
            mean_y = np.nanmean(y_idx_map, axis=1)
        x_rec = calculate_bar_position(mean_x, pitch)
        y_rec = calculate_bar_position(mean_y, pitch)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'argmax' or 'mean'.")

    if run_id is not None:
        rotation = get_run_hodo_rotation(run_id)
        if rotation is None:
            print(f"  No hodoscope rotation recorded for run {run_id} in run_list.json; "
                  f"using the unrotated bar frame.")
        else:
            x_rec, y_rec = rotate_to_tracker_frame(x_rec, y_rec, *rotation)

    return x_rec, y_rec, good_hodo
