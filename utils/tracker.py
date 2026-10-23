"""Loader for the raw silicon-tracker ASCII dumps (``TrackerData/by_run/<run>``).

One ``.dat`` file per spill, one line per event, 29 whitespace-separated
fields. Fields 0-5 are x/y for the 3 stations (only stations 1 and 2 are
exposed here -- station 3 is permanently unplugged, its y-column a
constant -6000 sentinel). Fields 18-28 are DAQ bookkeeping, per the
tracker team's own documentation:

    18. Nr of event within the spill
    19. Constant (0x50abcdef) -- a fixed sync/magic marker, not per-event data
    20. Timestamp silicon, 16 low bits   \\_ the tracker's own local clock
    21. Timestamp silicon, 8 upper bits  /  (not currently used here)
    22. Timestamp sent to DREAM, 16 low bits   \\_ combined into field 24
    23. Timestamp sent to DREAM, 8 upper bits  /
    24. Timestamp sent to DREAM, packed into one 24-bit hex value (==
        23:22 concatenated) -- this is what
        ``align_tracker_to_root_by_timestamp`` uses to find spill boundaries
    25. Nr of event from the start of the run + 1
    26. DREAM event number (observed all-zero in the runs checked so far)
    27. Nr of event within the spill (duplicate of field 18)
    28. Global event nr (same value as field 25)

Field 24 is "the timestamp *sent to* DREAM". The value DREAM received
doesn't survive the ROOT conversion as a per-event key (checked
``timestampbegin``/``timestampend``/``device_n``/``event_flag``, across
multiple runs -- all constant or all-zero placeholders), and field 26
(``dream_event_nr``) is zero in every official run. But DREAM keeps its own
microsecond clock (``FERS_Board*_tstamp_us``, see ``root_timestamp_us``),
and field 24 follows it on one straight line per run to ~0.3 us, while
triggers are ~1-8 ms apart -- so ``align_tracker_to_root_by_timestamp``
matches events by time alone, no event counting.
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from utils.constants import Z_H, Z_Si1, Z_Si2
from utils.io import project_root
from utils.plotting import get_runtype

TRACKER_DATA_ROOT = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/TrackerData/by_run"

# No-hit sentinel per station coordinate: -(column index + 1) * 1000, e.g.
# station 3's unplugged y-column sentinel of -6000 follows the same pattern.
SENTINEL_X1, SENTINEL_Y1, SENTINEL_X2, SENTINEL_Y2 = -1000.0, -2000.0, -3000.0, -4000.0
# mm-scale companions, for the *_mm calibrated branches build_aligned_tracker_branches
# adds when given a run_id -- see calibrate_tracker_positions below. Plain
# unit conversion, not calibration-scaled, so they stay exact/checkable.
SENTINEL_X1_MM, SENTINEL_Y1_MM = 10 * SENTINEL_X1, 10 * SENTINEL_Y1
SENTINEL_X2_MM, SENTINEL_Y2_MM = 10 * SENTINEL_X2, 10 * SENTINEL_Y2

# Straight-line weights putting the two-station track at the hodoscope's z
# (see scripts.hodores's module docstring): x_track = TRACK_W1 * x1 + TRACK_W2 * x2.
TRACK_W1 = (Z_H - Z_Si2) / (Z_Si1 - Z_Si2)  # -0.915
TRACK_W2 = (Z_Si1 - Z_H) / (Z_Si1 - Z_Si2)  # +1.915

# Column indices (0-based) into each whitespace-split line.
_COL_X1, _COL_Y1, _COL_X2, _COL_Y2 = 0, 1, 2, 3
_COL_EVENT_IN_SPILL = 18
_COL_TIMESTAMP_TO_DREAM = 24  # hex; "timestamp sent to DREAM", packed 24-bit
_COL_RUN_EVENT_NR = 25       # hex; "event from start of run + 1"
_COL_DREAM_EVENT_NR = 26     # hex; DREAM's own event counter
_COL_GLOBAL_EVENT_NR = 28    # decimal; same value as _COL_RUN_EVENT_NR

TRACKER_DTYPE = np.dtype([
    ("x1", "f8"), ("y1", "f8"),
    ("x2", "f8"), ("y2", "f8"),
    ("event_in_spill", "i8"),
    ("timestamp_to_dream", "i8"),
    ("run_event_nr", "i8"),
    ("dream_event_nr", "i8"),
    ("global_event_nr", "i8"),
])


def load_tracker_file(path: str | os.PathLike) -> np.ndarray:
    """Parse one spill's ``.dat`` file into a structured array (see ``TRACKER_DTYPE``).

    Some spills recorded zero events; those files are 0 bytes and yield an
    empty array rather than a warning from ``np.loadtxt``.
    """
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=TRACKER_DTYPE)

    hex16 = lambda s: int(s, 16)
    raw = np.loadtxt(
        path,
        usecols=(_COL_X1, _COL_Y1, _COL_X2, _COL_Y2,
                 _COL_EVENT_IN_SPILL, _COL_TIMESTAMP_TO_DREAM, _COL_RUN_EVENT_NR,
                 _COL_DREAM_EVENT_NR, _COL_GLOBAL_EVENT_NR),
        converters={
            _COL_EVENT_IN_SPILL: hex16,
            _COL_TIMESTAMP_TO_DREAM: hex16,
            _COL_RUN_EVENT_NR: hex16,
            _COL_DREAM_EVENT_NR: hex16,
        },
        ndmin=2,
    )
    out = np.empty(raw.shape[0], dtype=TRACKER_DTYPE)
    out["x1"], out["y1"], out["x2"], out["y2"] = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    out["event_in_spill"] = raw[:, 4]
    out["timestamp_to_dream"] = raw[:, 5]
    out["run_event_nr"] = raw[:, 6]
    out["dream_event_nr"] = raw[:, 7]
    out["global_event_nr"] = raw[:, 8]
    return out


def load_tracker_run(run_id: str | int, root: str | os.PathLike = TRACKER_DATA_ROOT) -> np.ndarray:
    """Concatenate every spill file for ``run_id`` (sorted by filename) into one array."""
    run_dir = os.path.join(root, str(run_id))
    files = sorted(glob.glob(os.path.join(run_dir, "*.dat")))
    if not files:
        raise FileNotFoundError(f"No tracker .dat files found in {run_dir}")
    return np.concatenate([load_tracker_file(f) for f in files])


def station1_hit_mask(data: np.ndarray) -> np.ndarray:
    """Events where station 1 actually recorded a hit (no sentinel)."""
    return (data["x1"] != SENTINEL_X1) & (data["y1"] != SENTINEL_Y1)


def station2_hit_mask(data: np.ndarray) -> np.ndarray:
    """Events where station 2 actually recorded a hit (no sentinel)."""
    return (data["x2"] != SENTINEL_X2) & (data["y2"] != SENTINEL_Y2)


def extrapolate_to_hodo(t1, t2):
    """Straight line through the two station hits, evaluated at the hodoscope's z (``Z_H``)."""
    return TRACK_W1 * np.asarray(t1) + TRACK_W2 * np.asarray(t2)


def _unwrap_counter(counter, modulus: int = 2 ** 31):
    """Undo overflow wraps in a free-running hardware counter.

    Once the counter exceeds its bit width it resets back down near zero,
    which looks like a huge *negative* diff (as opposed to the huge
    *positive* diff a real spill boundary produces) -- e.g. the tracker's
    raw timestamp counter has been observed wrapping around 2**31. Any
    negative diff at all is a wrap (the counter should otherwise only ever
    increase), so each one adds another multiple of ``modulus`` to
    everything from that point on, making the sequence monotonic again.
    """
    counter = np.asarray(counter, dtype=np.int64)
    if len(counter) < 2:
        return counter
    diffs = np.diff(counter)
    wraps = np.cumsum(diffs < 0)
    return counter + np.concatenate(([0], wraps)) * modulus


def find_spill_boundaries(counter, threshold_factor: float = 300, merge_gap: int = 3,
                          modulus: int = 2 ** 31):
    """Row indices marking spill boundaries, as ``(segment_end, next_start_minus_1)``.

    The beam turns off between spills, so any monotonic per-event clock
    (the tracker's raw hardware counter, or a ROOT timestamp branch) shows a
    gap there orders of magnitude larger than the typical inter-event
    spacing -- ``threshold_factor`` is applied to the *median* diff so this
    adapts to either clock's tick rate without needing to know it.

    A boundary is often not one clean jump but a short cluster of several
    large gaps within a few rows of each other -- confirmed directly on
    real data: 2-3 straggler rows (consistent with dead-time/afterglow
    triggers, not real in-spill activity) each preceded by its own huge
    gap, landing right at the edge of a spill. Collapsing a cluster to a
    single point and folding the stragglers into whichever segment that
    point lands in (the previous behavior, keeping only the last candidate)
    silently inflates that segment's real time span without a matching row
    count -- confirmed directly: this made the last ~1% of a segment's rows
    account for ~50% of its total elapsed time, identically shaped across
    several independent runs and beam energies, far too consistent to be
    beam-intensity physics. Instead, ``merge_gap`` groups nearby candidates
    into a cluster and the straggler rows *between* them are excluded from
    both segments: ``segment_end`` is the first candidate in the cluster
    (the last row of real dense activity), ``next_start_minus_1`` is the
    last candidate (the last row of the straggler zone) -- the following
    segment starts only after that. The two arrays are identical wherever a
    boundary was a single clean jump with no clustering.

    ``counter`` is unwrapped first (see ``_unwrap_counter``) in case an
    overflow happens to land right at a spill boundary -- otherwise that
    boundary's real jump would show up as a huge negative diff instead and
    be missed entirely, silently merging two spills into one segment.

    Returns
    -------
    segment_end : ndarray of int
    next_start_minus_1 : ndarray of int
    """
    counter = _unwrap_counter(counter, modulus)
    if len(counter) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    diffs = np.diff(counter)
    median = np.median(diffs)
    if median <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    candidates = np.where(diffs > median * threshold_factor)[0]
    if len(candidates) == 0:
        empty = candidates.astype(np.int64)
        return empty, empty.copy()

    clusters = [[candidates[0]]]
    for c in candidates[1:]:
        if c - clusters[-1][-1] <= merge_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    segment_end = np.array([cl[0] for cl in clusters], dtype=np.int64)
    next_start_minus_1 = np.array([cl[-1] for cl in clusters], dtype=np.int64)
    return segment_end, next_start_minus_1


def scan_alignment_correlation(x_tracker, root_idx, x_root, good_root,
                               shift_range: int = 15, min_n: int = 50):
    """QA check: is this run's alignment finding a real correspondence, or noise?

    ``match_frac`` (from ``align_tracker_to_root_by_timestamp``) only measures
    whether the algorithm found *some* self-consistent offset -- it says
    nothing about whether that offset is the *correct* one. Confirmed on
    real data this session: runs with match_frac near 100% ranged from a
    real, verifiable position correlation (e.g. run 1774, pi+ 60 GeV,
    r=0.40) down to no detectable correlation at all (e.g. run 1809, pi+
    160 GeV before excluding sentinels; run 1866, e+ 40 GeV, even after) --
    and this didn't track beam type, energy, or match_frac itself in any
    simple way. match_frac cannot be trusted alone to flag a bad run.

    This instead re-derives ``root_idx`` at small integer shifts and
    recomputes the raw (unbinned, per-event -- NOT the profile/binned
    correlation ``draw_fit`` plots, which can read as near-perfect even
    when the per-event correspondence is quite weak, since averaging
    ~100 events per profile bin hides exactly this kind of noise) Pearson
    correlation against an independent reference position (e.g. the
    hodoscope's) at each one. A real alignment should show a
    correlation peak at (or very near) shift 0, distinct from the shifts
    around it -- shifting to the wrong event should destroy any genuine
    per-event position correspondence. No distinguishable peak means the
    alignment likely isn't finding true correspondences for this run,
    whatever match_frac claims.

    The weak peaks quoted above (r ~0.4-0.8) came from the old
    counting-based matcher and turned out to be mismatched whole spills, not
    physics: with the time-coincidence matcher the same runs read r ~0.97-0.997
    against the hodoscope (see ``align_tracker_to_root_by_timestamp``).

    Parameters
    ----------
    x_tracker : tracker-side position array, already reduced to matched
        rows only (e.g. ``tracker["x1"][tracker_mask]``) -- filter out
        sentinel (no-hit) rows first, or a handful of extreme outliers
        will dominate and corrupt every ``r`` below (confirmed on real
        data: run 1809, 39 sentinel rows out of 70716 collapsed r from
        0.23 to 0.01).
    root_idx : root-side row index per matched tracker row, as returned by
        ``align_tracker_to_root_by_timestamp``
    x_root : the independent reference position array, full length
        (e.g. hodoscope x, indexed by ROOT event)
    good_root : bool mask, full length, marking which ``x_root`` entries
        are trustworthy (e.g. ``good_hodo``)
    shift_range : int, how far either side of 0 to scan
    min_n : int, a shift with fewer than this many valid pairs is skipped
        (reported as NaN) rather than trusted

    Returns
    -------
    list of (shift, r, n) tuples, one per shift in ``range(-shift_range,
    shift_range + 1)``; ``r`` is NaN where fewer than ``min_n`` pairs were
    available.
    """
    x_tracker = np.asarray(x_tracker)
    root_idx = np.asarray(root_idx)
    x_root = np.asarray(x_root)
    good_root = np.asarray(good_root)
    n_root = len(x_root)

    results = []
    for shift in range(-shift_range, shift_range + 1):
        shifted = root_idx + shift
        valid = (shifted >= 0) & (shifted < n_root)
        idx_v = shifted[valid]
        x_v = x_tracker[valid]
        keep = good_root[idx_v]
        n = int(keep.sum())
        if n < min_n:
            results.append((shift, float("nan"), n))
            continue
        r = float(np.corrcoef(x_v[keep], x_root[idx_v][keep])[0, 1])
        results.append((shift, r, n))
    return results


def root_timestamp_us(root_tstamp_us):
    """Exact DREAM event time (int64 us) from a ``FERS_Board*_tstamp_us`` branch.

    The ROOT conversion stores the 32-bit microsecond timestamp T as the
    64-bit word ``(T << 32) | T`` cast to float64 (``FERS_Board*_trigger_id``
    is packed the same way: n * 2**32 + n). A float64 keeps 53 bits, so the
    upper copy survives exactly while the lower one is rounded -- to 4 us
    early in a run, 1-2 ms late in it. This module used to read the lower
    word, which left neighbouring triggers (~1-8 ms apart) indistinguishable
    by time. Raises if the two copies disagree by more than that rounding,
    i.e. if a branch isn't packed this way.
    """
    u = np.asarray(root_tstamp_us).astype(np.uint64)
    upper = (u >> np.uint64(32)).astype(np.int64)
    lower = (u & np.uint64(0xFFFFFFFF)).astype(np.int64)
    if len(u) and np.max(np.abs(upper - lower)) > 4096:
        raise ValueError("tstamp_us isn't packed as (T << 32) | T; can't decode the DREAM timestamp.")
    return upper


# DREAM microseconds per tracker ``timestamp_to_dream`` tick (2.5 ticks/us),
# 0.3999977-0.3999981 per spill on every run checked. Only seeds a spill's
# offset search -- the slope is refitted per spill -- where a 5e-7 error is
# ~6 us over a ~5 s spill, well inside the search window.
TRACKER_CLOCK_SLOPE = 0.399998


def _nearest(sorted_times, x):
    """Index of, and signed distance to, the nearest ``sorted_times`` entry for each ``x``."""
    j = np.clip(np.searchsorted(sorted_times, x), 1, len(sorted_times) - 1)
    j = np.where(x - sorted_times[j - 1] <= sorted_times[j] - x, j - 1, j)
    return j, sorted_times[j] - x


def _fit_spill_clock(t_spill, root_us, guess, rng, slope0, search_window_us, match_window_us,
                     n_probe, min_peak_frac):
    """``root_us = slope * tracker_ticks + intercept`` for one tracker spill, by time coincidence.

    ``guess`` is the previous spill's ``(slope, intercept)``, or None. The
    clock relationship is one line across a whole run (the intercept drifts
    ~1 us per spill), so it's tried first. Failing that, ``n_probe`` random
    rows each vote for the intercept that would put them on every ROOT
    event, at ``slope0``: only the true intercept collects (nearly) all of
    them -- chance coincidences are ~1e-3 per probe in a 20 us window. The
    slope and intercept are then refitted on the coincident rows as the
    window tightens to ``match_window_us``.

    Returns ``(slope, intercept, peak_frac)`` -- ``peak_frac`` the fraction
    of probe rows that coincided -- or None if under ``min_peak_frac`` did.
    """
    probe = t_spill[rng.choice(len(t_spill), min(n_probe, len(t_spill)), replace=False)]
    found = None
    if guess is not None:
        _, d = _nearest(root_us, guess[0] * probe + guess[1])
        hit = np.abs(d) < search_window_us
        if hit.mean() >= min_peak_frac:
            found = guess[0], guess[1] + np.median(d[hit]), hit.mean()
    if found is None:
        votes = np.sort((root_us[None, :] - slope0 * probe[:, None]).ravel())
        in_window = np.searchsorted(votes, votes + search_window_us) - np.arange(len(votes))
        k = int(np.argmax(in_window))
        if in_window[k] < min_peak_frac * len(probe):
            return None
        found = slope0, votes[k] + search_window_us / 2, in_window[k] / len(probe)

    slope, intercept, peak_frac = found
    for window in (search_window_us, 5.0, match_window_us):
        j, d = _nearest(root_us, slope * t_spill + intercept)
        ok = np.abs(d) < window
        if ok.sum() < 2:
            return None
        slope, intercept = np.polyfit(t_spill[ok], root_us[j[ok]], 1)
    return slope, intercept, peak_frac


def _time_based_segment_match(t_raw_seg, r_raw_seg, slope, intercept, max_residual=None):
    """Nearest-neighbor match in real (fitted) time within one segment.

    Strictly monotonic by construction (``j`` only ever advances): once a
    root row is used, no later tracker row can be assigned it again. This
    matters for the same reason it did for the counting-based walk --
    without it, a run of closely-spaced predictions can collapse onto the
    same root row and silently duplicate matches (confirmed: the first,
    unconstrained version of this function did exactly that).

    Parameters
    ----------
    max_residual : float, optional -- if given, a tracker row whose nearest
        root row is farther than this (in ``r_raw_seg`` units) is rejected
        (``matched=False``) rather than force-matched to it anyway. ``j``
        does not advance past a rejected row's root row: a tracker trigger
        DREAM never recorded is nearest to the *next* trigger's root row, and
        consuming it cost that next row its real match (confirmed: 0-11 rows
        per run, 0.2 us from their partner, before this was fixed). Omit (or
        None) to keep the old always-match behavior.

    Returns
    -------
    matched : ndarray of bool, shape (len(t_raw_seg),) -- True where the
        row was matched within ``max_residual`` (or always True, if
        ``max_residual`` is None); callers wanting a distance-based reject
        with no ``max_residual`` given can still check the residual
        themselves.
    local_root_idx : ndarray of int, position within ``r_raw_seg``
    """
    t_pred = slope * t_raw_seg.astype(np.float64) + intercept
    n_t, n_r = len(t_pred), len(r_raw_seg)
    matched = np.ones(n_t, dtype=bool)
    local_root_idx = np.full(n_t, -1, dtype=np.int64)

    j = 0
    for i in range(n_t):
        if j >= n_r:
            matched[i:] = False
            break
        tt = t_pred[i]
        while j + 1 < n_r and abs(r_raw_seg[j + 1] - tt) <= abs(r_raw_seg[j] - tt):
            j += 1
        local_root_idx[i] = j
        if max_residual is not None and abs(r_raw_seg[j] - tt) > max_residual:
            matched[i] = False
            continue  # leave root row j for the next tracker row
        j += 1  # strictly increasing -- guarantees no root row is reused

    return matched, local_root_idx


# scan_alignment_correlation's shift=0 real correlation, below which an
# alignment that otherwise passes match_frac is refused. Picked from real
# data (e+ 40 GeV, runs 1846-1882): garbage alignments (confirmed via this
# same check to be mismatching essentially every event) topped out at
# |r|=0.02, while genuine alignments started at r=0.39 -- 0.2 sits in the
# middle of that gap with margin on both sides. match_frac alone isn't
# enough: it only confirms the algorithm found *some* self-consistent
# offset, not that the offset is correct (confirmed: match_frac 0.69-0.84,
# comfortably above min_match_frac, on runs whose real correlation was
# indistinguishable from zero -- across 23 same-energy runs checked this
# way, real correlation strength predicted measured si-tracker efficiency
# far better than match_frac did, Pearson r=0.92 vs 0.87). Those numbers
# are from the old counting-based matcher; with time coincidence genuine
# runs read r ~0.97+, so this is now only a backstop.
MIN_ALIGNMENT_CORRELATION = 0.2


def align_tracker_to_root_by_timestamp(tracker: np.ndarray, trigger_n, root_tstamp_us,
                                      threshold_factor: float = 300, merge_gap: int = 3,
                                      clock_slope: float = TRACKER_CLOCK_SLOPE,
                                      search_window_us: float = 20.0, match_window_us: float = 3.0,
                                      n_probe: int = 50, min_peak_frac: float = 0.5,
                                      min_spill_rows: int = 10, min_match_frac: float = 0.5,
                                      x_root=None, good_root=None,
                                      min_correlation: float = MIN_ALIGNMENT_CORRELATION):
    """Match tracker events to ROOT events by their clock times.

    The tracker's ``timestamp_to_dream`` and DREAM's microsecond clock
    (``root_timestamp_us``) follow one straight line per run to ~0.3 us RMS,
    while triggers are ~1-8 ms apart, so a tracker event's counterpart is
    just the ROOT event at its predicted time. Per tracker spill
    (``find_spill_boundaries``; ROOT's own spills aren't needed):
    ``_fit_spill_clock`` finds the line by time coincidence, then each row
    is matched to its nearest ROOT event within ``match_window_us``
    (``_time_based_segment_match``). Rows with no ROOT event that close stay
    unmatched. ``trigger_n`` is unused, kept for callers.

    Replaced (2026-10-15) a matcher that paired tracker and ROOT spills by
    their boundaries, took each spill's offset from counting events, and
    read the rounded lower timestamp word (see ``root_timestamp_us``). Its
    errors came in whole spills: a count off by a few events shifted every
    match in the spill, and the straight-line clock fit still looked
    perfect. Checked against the hodoscope (|hodo - two-station track| <
    1 mm in x and y), it left 17-36% of matched events inconsistent on runs
    1774/1813/1832 and ~100% on 1765/1866; this leaves 1.3-4%, with ~100% of
    tracker rows matched and no ROOT event used twice. At 0.3 us timing the
    remainder isn't mismatches -- it's highest at e+ 40 GeV, so most likely
    scattering/showering tails.

    Parameters
    ----------
    clock_slope : seeds each spill's offset search (``_fit_spill_clock``).
    search_window_us : coincidence window for finding a spill's offset.
    match_window_us : a row farther than this from every ROOT event (after
        the spill's refit) stays unmatched.
    n_probe, min_peak_frac : rows sampled per spill to find its offset, and
        the fraction of them that must coincide for the spill to be matched
        at all.
    min_spill_rows : smaller tracker "spills" (straggler clusters) are skipped.
    min_match_frac : refuse a run where fewer tracker rows than this match.
    x_root, good_root : optional -- an independent reference position (e.g.
        hodoscope x) and its goodness mask, both full ROOT-event length.
        When given, the result is additionally checked against
        ``scan_alignment_correlation`` at shift 0 and rejected if that real
        per-event correlation falls below ``min_correlation`` -- see
        ``MIN_ALIGNMENT_CORRELATION`` for why match_frac alone isn't a
        strong enough gate. Omit to skip this check (e.g. for callers with
        no independent reference available).

    Returns
    -------
    tracker_mask : ndarray of bool, shape (len(tracker),)
    root_idx : ndarray of int, one entry per True in ``tracker_mask``
    fit_diagnostics : list of (t_lo, t_hi, fit_or_None) per tracker spill,
        ``fit_or_None`` being ``(slope, intercept, peak_frac, n_matched,
        resid_rms_us)`` or ``None`` for a spill whose offset wasn't found.
    match_frac : float
    """
    root_us = _unwrap_counter(root_timestamp_us(root_tstamp_us), modulus=2 ** 32).astype(np.float64)
    t_raw = _unwrap_counter(tracker["timestamp_to_dream"], modulus=2 ** 31).astype(np.float64)
    ends, next_starts = find_spill_boundaries(tracker["timestamp_to_dream"], threshold_factor,
                                              merge_gap, modulus=2 ** 31)
    starts = np.concatenate(([0], next_starts + 1))
    stops = np.concatenate((ends + 1, [len(tracker)]))

    rng = np.random.default_rng(0)  # picks probe rows only; fixed so a run always aligns the same
    tracker_mask = np.zeros(len(tracker), dtype=bool)
    root_idx_parts = []
    fit_diagnostics = []
    guess = None
    for t_lo, t_hi in zip(starts, stops):
        if t_hi - t_lo < min_spill_rows:
            continue
        t_spill = t_raw[t_lo:t_hi]
        fit = _fit_spill_clock(t_spill, root_us, guess, rng, clock_slope, search_window_us,
                               match_window_us, n_probe, min_peak_frac)
        if fit is None:
            fit_diagnostics.append((int(t_lo), int(t_hi), None))
            continue
        slope, intercept, peak_frac = fit
        guess = slope, intercept

        pred = slope * t_spill + intercept
        r_lo = max(int(np.searchsorted(root_us, pred[0] - search_window_us)) - 1, 0)
        r_hi = int(np.searchsorted(root_us, pred[-1] + search_window_us)) + 1
        matched, local_idx = _time_based_segment_match(t_spill, root_us[r_lo:r_hi], slope, intercept,
                                                       match_window_us)
        resid = root_us[r_lo + local_idx[matched]] - pred[matched]
        resid_rms = float(np.sqrt(np.mean(resid ** 2))) if matched.any() else float("nan")
        fit_diagnostics.append((int(t_lo), int(t_hi), (slope, intercept, peak_frac, int(matched.sum()), resid_rms)))
        tracker_mask[t_lo + np.flatnonzero(matched)] = True
        root_idx_parts.append(r_lo + local_idx[matched])

    root_idx = np.concatenate(root_idx_parts) if root_idx_parts else np.empty(0, dtype=np.int64)
    match_frac = tracker_mask.sum() / len(tracker) if len(tracker) else 0.0
    if match_frac < min_match_frac:
        raise ValueError(
            f"Only {match_frac:.1%} of tracker events matched a ROOT event in time; "
            f"refusing to align (min_match_frac={min_match_frac:.1%})."
        )

    if x_root is not None and good_root is not None:
        x_tracker_all = tracker["x1"][tracker_mask]
        keep = x_tracker_all != SENTINEL_X1
        corr = scan_alignment_correlation(x_tracker_all[keep], root_idx[keep], x_root, good_root,
                                          shift_range=0)
        r0 = corr[0][1]
        if not (r0 == r0) or r0 < min_correlation:  # nan-safe (r0 == r0 is False for NaN)
            raise ValueError(
                f"Real per-event correlation at shift 0 is {r0:.3f} (< {min_correlation}); "
                f"alignment passes match_frac ({match_frac:.1%}) but doesn't find true "
                f"tracker<->hodoscope correspondences, refusing."
            )

    return tracker_mask, root_idx, fit_diagnostics, match_frac


def root_matched_mask(tracker: np.ndarray, trigger_n, root_tstamp_us, n_root=None, **kwargs):
    """Boolean mask over every ROOT event: True where it has a genuine
    tracker counterpart, per ``align_tracker_to_root_by_timestamp``.

    This is the direct answer to "which ROOT rows have no matching tracker
    data" -- ``~mask`` marks ROOT events to drop for any analysis that
    needs a real tracker hit, the same way ``tracker_matched`` does in
    ``build_aligned_tracker_branches``.

    Parameters
    ----------
    n_root : int, optional -- length of the ROOT event array, if not
        simply ``len(trigger_n)`` (e.g. already filtered/sliced elsewhere).
    **kwargs : passed straight through to ``align_tracker_to_root_by_timestamp``

    Returns
    -------
    root_matched : ndarray of bool, shape (n_root,)
    match_frac : float, tracker-side match fraction (see
        ``align_tracker_to_root_by_timestamp``) -- diagnostic, not the same
        thing as ``root_matched.mean()``
    """
    trigger_n = np.asarray(trigger_n)
    if n_root is None:
        n_root = len(trigger_n)

    tracker_mask, root_idx, fit_diagnostics, match_frac = align_tracker_to_root_by_timestamp(
        tracker, trigger_n, root_tstamp_us, **kwargs)

    root_matched = np.zeros(n_root, dtype=bool)
    root_matched[root_idx] = True
    return root_matched, match_frac


# ---------------------------------------------------------------------------
# Tracker -> hodoscope frame calibration
# ---------------------------------------------------------------------------
#
# ``scripts/sihodocor.py --testbeam`` fits, per run, station and axis,
# tracker_mm = slope * hodo_mm + intercept on veto-selected, time-matched
# events (the hodoscope already rotated into the tracker frame by
# ``reconstruct_hodoscope(run_id=...)``). ``calibrate_tracker_positions``
# inverts it, (tracker_mm - intercept) / slope, so each station reads in the
# hodoscope's own coordinates: its HodoX-vs-Tracker plot fits y = x.
# build_aligned_tracker_branches (below) applies it when given a ``run_id``.
#
# The slope is not a detector scale: it is the beam's position-angle
# correlation projected over each station's gap to the hodoscope (Si1, 90 cm
# upstream, departs from 1 about twice as much as Si2, 43 cm), so it moves
# with the beam optics, 0.98-1.07 across TB2026. That makes the calibrated
# positions right for single-station use -- each station's on-average
# estimate of where the particle crosses the hodoscope -- but never build a
# two-station track from them: a track needs the raw scale (see
# scripts/hodores.py). The intercept is geometry (the two detectors'
# unsurveyed origins, ~49-56 mm) and holds to ~1 mm across TB2026, table
# moves included.

# Minimum |Pearson r| / profile points for a run's own fitted Hodo-vs-Tracker
# slope (from the calibration CSV below) to be trusted rather than noise --
# see scripts/sihodocor.py's fit_profile_line docstring: a straight line
# through exactly 2 points has |r| = 1 by construction regardless of
# correctness, so the point-count cut is needed alongside the r cut, not
# instead of it.
TRACKER_HODO_MIN_FIT_R = 0.8
TRACKER_HODO_MIN_FIT_POINTS = 4

TRACKER_HODO_KEYS = [("Tracker1", "x"), ("Tracker1", "y"), ("Tracker2", "x"), ("Tracker2", "y")]

# Versioned (not gitignored, unlike output/) copy of each testbeam's
# ``scripts.sihodocor --testbeam`` bigplot CSV -- this is what makes that
# CSV an explicit calibration *input* to the reconstruction rather than a
# transient artifact that happens to still be sitting in output/ from a
# previous run. Regenerate/update via:
#   python -m scripts.sihodocor --testbeam <name>
#   cp output/sihodocor/bigplot/<name>_slopes.csv data/tracker_hodo_slopes/
TRACKER_HODO_SLOPES_DIR = os.path.join(project_root(), "data", "tracker_hodo_slopes")


def get_tracker_hodo_calibration(run_id):
    """Return this run's {("Tracker1"|"Tracker2", "x"|"y"): (slope, intercept_mm)}.

    Prefers run_id's own row in
    ``data/tracker_hodo_slopes/<testbeam>_slopes.csv`` (quality-gated by
    TRACKER_HODO_MIN_FIT_R / TRACKER_HODO_MIN_FIT_POINTS -- a run's own fit
    can itself be a degenerate/garbage outlier, e.g. a near-zero-statistics
    run reduced to a 2-point fit). A key without a trusted fit of its own
    takes the median of the testbeam's trusted fits. Empty dict if the
    testbeam has no CSV. A CSV written before the intercept column
    (2026-10-15) reads as intercept 0, i.e. scale only.
    """
    runtype = get_runtype(run_id)
    csv_path = os.path.join(TRACKER_HODO_SLOPES_DIR, f"{runtype.replace(' ', '_')}_slopes.csv")
    if not os.path.exists(csv_path):
        return {}

    trusted = {}  # run -> {key: (slope, intercept)}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "tracker1_x_intercept" not in (reader.fieldnames or []):
            print(f"  {csv_path} has no intercepts (written before 2026-10-15); applying slopes only -- "
                  f"regenerate it with scripts.sihodocor --testbeam {runtype}.")
        for row in reader:
            for station, axis in TRACKER_HODO_KEYS:
                prefix = f"tracker{station[-1]}_{axis}"
                slope, r, n_points = row.get(f"{prefix}_slope"), row.get(f"{prefix}_r"), row.get(f"{prefix}_n_points")
                if not (slope and r and n_points):
                    continue
                if abs(float(r)) >= TRACKER_HODO_MIN_FIT_R and int(n_points) >= TRACKER_HODO_MIN_FIT_POINTS:
                    trusted.setdefault(row["run"], {})[(station, axis)] = (
                        float(slope), float(row.get(f"{prefix}_intercept") or 0.0))

    own = trusted.get(str(int(run_id)), {})
    calib = {}
    for key in TRACKER_HODO_KEYS:
        if key in own:
            calib[key] = own[key]
        else:
            fits = [run_fits[key] for run_fits in trusted.values() if key in run_fits]
            if fits:
                calib[key] = tuple(float(v) for v in np.median(fits, axis=0))
    if calib:
        source = ("this run's own fit" if len(own) == len(TRACKER_HODO_KEYS)
                  else f"this run's own fit for {sorted(own)}, testbeam median for the rest" if own
                  else "testbeam median (run not found/not trusted in the calibration CSV)")
        print(f"  Tracker/hodo calibration ({runtype}): {source}")
    return calib


def calibrate_tracker_positions(x1, y1, x2, y2, run_id):
    """Convert raw tracker positions (cm, as from load_tracker_run /
    build_aligned_tracker_branches) into the hodoscope's frame, in mm:
    (10 * x - intercept) / slope per station and axis (see
    get_tracker_hodo_calibration and the section comment above -- fine for
    single-station use, not for building tracks).

    No-hit sentinel rows (SENTINEL_X1 etc) are converted to their exact
    mm-scale companion (SENTINEL_X1_MM etc) rather than being calibrated --
    a sentinel is a flag, not a physical position, so transforming it would
    both be meaningless and silently break a downstream
    ``!= SENTINEL_X1_MM`` check. Returns (x1_mm, y1_mm, x2_mm, y2_mm); if no
    calibration is known for this run's testbeam, positions are still
    converted cm -> mm but not calibrated (a printed warning notes this).
    """
    x1, y1, x2, y2 = (np.asarray(a, dtype=float) for a in (x1, y1, x2, y2))
    calib = get_tracker_hodo_calibration(run_id)
    if not calib:
        print(f"  No tracker/hodo calibration known for testbeam {get_runtype(run_id)!r}; "
              f"using uncalibrated tracker mm.")

    out = []
    for x, sentinel_raw, sentinel_mm, key in [
        (x1, SENTINEL_X1, SENTINEL_X1_MM, ("Tracker1", "x")),
        (y1, SENTINEL_Y1, SENTINEL_Y1_MM, ("Tracker1", "y")),
        (x2, SENTINEL_X2, SENTINEL_X2_MM, ("Tracker2", "x")),
        (y2, SENTINEL_Y2, SENTINEL_Y2_MM, ("Tracker2", "y")),
    ]:
        slope, intercept = calib.get(key, (1.0, 0.0))
        hit = x != sentinel_raw
        x_mm = np.where(hit, (10 * x - intercept) / slope, sentinel_mm)
        out.append(x_mm)
    return tuple(out)


def build_aligned_tracker_branches(tracker: np.ndarray, trigger_n, root_tstamp_us,
                                   run_id=None, **align_kwargs):
    """Re-index tracker x1/y1/x2/y2 onto DREAM's own event ordering.

    Returns one row per ROOT event (length == len(trigger_n)), so it can be
    written straight into a companion ROOT file (see
    ``scripts/build_tracker_root.py``) or used as-is: a ROOT event with no
    trustworthy tracker row gets the tracker's own existing no-hit
    sentinels (``SENTINEL_X1`` etc, see ``station1_hit_mask`` /
    ``station2_hit_mask``), rather than being silently dropped -- so a
    plain ``!= SENTINEL_X1`` check downstream replaces the whole alignment
    pipeline for anyone who just wants the positions. Matches via
    ``align_tracker_to_root_by_timestamp`` (real clock time, see that
    function's docstring for how it works and why).

    Passing ``run_id`` additionally calibrates the positions into the
    hodoscope's frame, in mm (see ``calibrate_tracker_positions``)
    and adds them as ``tracker_x1_mm`` etc, alongside the untouched raw cm
    branches -- this is the reconstruction-time calibration step, so any
    caller that wants calibrated positions (e.g. ``scripts.tracker_shapefit``,
    or the companion ROOT file ``scripts.build_tracker_root`` writes) just
    passes ``run_id`` instead of re-deriving the correction itself.
    ``run_id=None`` (the default) skips this and returns exactly what this
    function always has, unchanged.

    Parameters
    ----------
    run_id : optional -- if given, adds calibrated ``*_mm`` branches (see above).
    **align_kwargs : passed straight through to align_tracker_to_root_by_timestamp

    Returns
    -------
    branches : dict of str -> ndarray, each shape (len(trigger_n),):
        ``tracker_x1``, ``tracker_y1``, ``tracker_x2``, ``tracker_y2``
        (sentinel-filled, raw cm), plus ``tracker_matched``, and -- only
        when ``run_id`` is given -- ``tracker_x1_mm``, ``tracker_y1_mm``,
        ``tracker_x2_mm``, ``tracker_y2_mm`` (sentinel-filled with
        ``SENTINEL_X1_MM`` etc, calibrated mm).
    match_frac : float, as returned by align_tracker_to_root_by_timestamp
    """
    trigger_n = np.asarray(trigger_n)
    n_root = len(trigger_n)

    tracker_mask, root_idx, fit_diagnostics, match_frac = align_tracker_to_root_by_timestamp(
        tracker, trigger_n, root_tstamp_us, **align_kwargs)
    tracker_aligned = tracker[tracker_mask]

    x1 = np.full(n_root, SENTINEL_X1)
    y1 = np.full(n_root, SENTINEL_Y1)
    x2 = np.full(n_root, SENTINEL_X2)
    y2 = np.full(n_root, SENTINEL_Y2)
    matched = np.zeros(n_root, dtype=bool)

    x1[root_idx] = tracker_aligned["x1"]
    y1[root_idx] = tracker_aligned["y1"]
    x2[root_idx] = tracker_aligned["x2"]
    y2[root_idx] = tracker_aligned["y2"]
    matched[root_idx] = True

    branches = {
        "tracker_x1": x1, "tracker_y1": y1,
        "tracker_x2": x2, "tracker_y2": y2,
        "tracker_matched": matched,
    }
    if run_id is not None:
        x1_mm, y1_mm, x2_mm, y2_mm = calibrate_tracker_positions(x1, y1, x2, y2, run_id)
        branches.update({
            "tracker_x1_mm": x1_mm, "tracker_y1_mm": y1_mm,
            "tracker_x2_mm": x2_mm, "tracker_y2_mm": y2_mm,
        })
    return branches, match_frac
