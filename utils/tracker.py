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

Field 24 is "the timestamp *sent to* DREAM", not an independent
tracker-only clock -- in principle DREAM should have recorded that same
value somewhere, which would let events be matched by exact equality
instead of the statistical alignment this module actually has to do. That
received value doesn't appear to survive the ROOT conversion into any
accessible branch (checked ``timestampbegin``/``timestampend``/``device_n``/
``event_flag``, across multiple runs -- all constant or all-zero
placeholders), so ``align_tracker_to_root_by_timestamp`` remains
inference, not an exact lookup. Same story for field 26 (``dream_event_nr``):
genuinely populated, but only in a handful of early pre-TB2026 runs, always
zero in every official run since.

``align_tracker_to_root_by_timestamp`` calibrates its per-segment clock fit
against ``trigger_n``, not ``event_n``, for the same reason: ``event_n`` is
a perfectly dense 0..N-1 range with zero gaps, so *any* candidate offset
that keeps shifted values in-bounds "matches" 100% of the time regardless
of whether it's the right one -- it can't discriminate at all. ``trigger_n``
has a handful of real gaps, which at least gives that calibration something
to constrain against (a counting-only approach was tried first and
abandoned -- see that function's docstring for why real clock time works
better than counting events).
"""

from __future__ import annotations

import glob
import os

import numpy as np

TRACKER_DATA_ROOT = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/TrackerData/by_run"

# No-hit sentinel per station coordinate: -(column index + 1) * 1000, e.g.
# station 3's unplugged y-column sentinel of -6000 follows the same pattern.
SENTINEL_X1, SENTINEL_Y1, SENTINEL_X2, SENTINEL_Y2 = -1000.0, -2000.0, -3000.0, -4000.0

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


def _match_boundary_offset(tracker_boundaries, root_boundaries,
                           max_shift: int = 10, min_segments: int = 5):
    """Find which (tracker boundary, root boundary) pair to start counting from.

    Tries small relative shifts at the start of each boundary list (a
    boundary right at the very start/end of a run can be caught by one
    clock and missed by the other) and scores each candidate alignment by
    how *consistent* the ratio of inter-boundary event counts is across
    segments -- root records a different, but roughly constant, fraction
    more triggers per spill than the tracker, so the true alignment has a
    low-scatter ratio while a wrong shift looks essentially random.

    Returns ``(t_start, r_start, n_segments, score)`` for the best
    candidate, or ``None`` if there aren't enough boundaries in common to
    judge confidently.
    """
    t_counts = np.diff(tracker_boundaries).astype(float)
    r_counts = np.diff(root_boundaries).astype(float)
    best = None
    for t_off in range(max(len(t_counts) - min_segments + 1, 0)):
        if t_off > max_shift:
            break
        for r_off in range(max(len(r_counts) - min_segments + 1, 0)):
            if r_off > max_shift:
                break
            n = min(len(t_counts) - t_off, len(r_counts) - r_off)
            if n < min_segments:
                continue
            ratio = r_counts[r_off:r_off + n] / t_counts[t_off:t_off + n]
            score = float(np.std(ratio) / np.mean(ratio))
            if best is None or score < best[3]:
                best = (t_off, r_off, n, score)
    return best


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

    A weak-but-real peak at shift 0 does not by itself prove the
    alignment is imprecise, either: some of the gap between this and the
    tracker's own internal station1-vs-station2 correlation (~0.98, since
    those two columns come from the same raw row and need no alignment at
    all) is plausibly genuine physics -- multiple scattering between the
    hodoscope and tracker planes -- not a matching error. This function
    reports the data; it does not attempt to separate those two causes.

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


def _fit_segment_time_calibration(run_event_nr, root_key_local, t_raw_seg, r_raw_seg,
                                  local_search_range: int = 200, min_anchor_count: int = 10,
                                  min_fit_r: float = 0.999):
    """Fit ``root_raw_ticks = slope * tracker_raw_ticks + intercept`` for one
    segment, using the existing counting-based match as a source of trusted
    anchor points -- not as the final answer, just to calibrate a real
    physical clock relationship between the two streams.

    Confirmed on real data (run 1774, run 1832): a genuine spill segment
    fits this line essentially perfectly (r >= 0.999999, slope landing
    almost exactly on 0.4 -- i.e. the tracker's clock runs at 2.5 ticks per
    root-microsecond -- consistently across dozens of independent segments
    in both runs). A segment that *isn't* real, continuous spill activity
    (confirmed: run 1832's very first segment, a long beam-off dead period
    with at most a couple of genuine events, where the counting-based
    "anchors" are mostly spurious ties rather than real matches) fits
    badly (r=0.41 in that case) -- ``min_fit_r`` rejects segments like that
    rather than building a matching scheme on a meaningless fit.

    Returns
    -------
    (slope, intercept, r, n_anchors) or None if there aren't enough anchor
    points, or the fit quality doesn't clear ``min_fit_r``.
    """
    naive_offset = int(run_event_nr.min() - root_key_local.min())
    best_offset, best_n = naive_offset, -1
    for delta in range(-local_search_range, local_search_range + 1):
        offset = naive_offset + delta
        cnt = np.isin(run_event_nr - offset, root_key_local).sum()
        if cnt > best_n:
            best_offset, best_n = offset, cnt

    anchor_mask = np.isin(run_event_nr - best_offset, root_key_local)
    if anchor_mask.sum() < min_anchor_count:
        return None

    anchor_i = np.where(anchor_mask)[0]
    lookup = {}
    for idx, v in enumerate(root_key_local):
        lookup.setdefault(v, idx)
    anchor_j = np.array([lookup[run_event_nr[i] - best_offset] for i in anchor_i])

    t_anchor = t_raw_seg[anchor_i].astype(np.float64)
    r_anchor = r_raw_seg[anchor_j].astype(np.float64)
    slope, intercept = np.polyfit(t_anchor, r_anchor, 1)
    r_corr = float(np.corrcoef(t_anchor, r_anchor)[0, 1])
    if r_corr < min_fit_r:
        return None
    return slope, intercept, r_corr, int(anchor_mask.sum())


def _time_based_segment_match(t_raw_seg, r_raw_seg, slope, intercept):
    """Nearest-neighbor match in real (fitted) time within one segment.

    Strictly monotonic by construction (``j`` only ever advances): once a
    root row is used, no later tracker row can be assigned it again. This
    matters for the same reason it did for the counting-based walk --
    without it, a run of closely-spaced predictions can collapse onto the
    same root row and silently duplicate matches (confirmed: the first,
    unconstrained version of this function did exactly that).

    Returns
    -------
    matched : ndarray of bool, shape (len(t_raw_seg),) -- always True here
        (every tracker row gets *some* nearest match); callers wanting a
        distance-based reject should check the residual themselves.
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
        j += 1  # strictly increasing -- guarantees no root row is reused

    return matched, local_root_idx


def align_tracker_to_root_by_timestamp(tracker: np.ndarray, trigger_n, root_tstamp_us,
                                      threshold_factor: float = 300, merge_gap: int = 3,
                                      max_shift: int = 10, min_segments: int = 5,
                                      max_ratio_cv: float = 0.1, local_search_range: int = 200,
                                      min_anchor_count: int = 10, min_fit_r: float = 0.999,
                                      min_match_frac: float = 0.5):
    """Match tracker events to ROOT by real physical time, not by counting.

    Finds spill boundaries independently in each stream (see
    ``find_spill_boundaries`` / ``_match_boundary_offset``), then within
    each matched segment fits an actual clock relationship (``root_ticks =
    slope * tracker_ticks + intercept``, see
    ``_fit_segment_time_calibration``) using a brute-force count-based
    match only as a calibration aid to find trusted anchor points, then
    matches every tracker row to its nearest root row in that fitted real
    time -- not by counting at all.

    An earlier version of this module matched purely by counting
    (walking ``run_event_nr``/``trigger_n`` row by row, guessing which
    side had dropped an event whenever the running offset didn't line up).
    That approach is gone now: confirmed directly on real data that
    matching by real time does measurably better on every run tested (raw
    per-event position correlation 0.66-0.81 under counting -> 0.74-0.82
    under time fitting), because the underlying clock relationship is
    extremely stable (r >= 0.999999 in nearly every segment tested) --
    counting has to guess where individual events were dropped; time
    doesn't need to guess, it just needs an accurate clock.

    A segment whose fit doesn't clear ``min_fit_r`` (see
    ``_fit_segment_time_calibration`` -- confirmed to reliably flag
    segments that aren't real continuous spill activity, e.g. a long
    beam-off dead period with only a couple of genuine events) is skipped
    entirely rather than matched on a meaningless calibration.

    Returns
    -------
    tracker_mask : ndarray of bool, shape (len(tracker),)
    root_idx : ndarray of int, one entry per True in ``tracker_mask``
    fit_diagnostics : list of (t_lo, t_hi, fit_or_None) per segment, where
        ``fit_or_None`` is ``(slope, intercept, r, n_anchors)`` or ``None``
        for a rejected segment -- for inspecting which parts of a run this
        trusted enough to match at all.
    match_frac : float
    """
    trigger_n = np.asarray(trigger_n)
    root_ts = (np.asarray(root_tstamp_us).astype(np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.int64)
    t_raw = _unwrap_counter(tracker["timestamp_to_dream"], modulus=2 ** 31)
    r_raw = _unwrap_counter(root_ts, modulus=2 ** 32)

    t_boundaries, t_next_start = find_spill_boundaries(tracker["timestamp_to_dream"], threshold_factor,
                                                       merge_gap, modulus=2 ** 31)
    r_boundaries, r_next_start = find_spill_boundaries(root_ts, threshold_factor, merge_gap,
                                                       modulus=2 ** 32)
    match = _match_boundary_offset(t_boundaries, r_boundaries, max_shift, min_segments)
    if match is None:
        raise ValueError(
            f"Not enough spill boundaries to align by time fit "
            f"(tracker: {len(t_boundaries)}, root: {len(r_boundaries)}; need >= {min_segments} segments)."
        )
    t_off, r_off, n, score = match
    if score > max_ratio_cv:
        raise ValueError(
            f"Best spill-boundary match is still inconsistent (cv={score:.3f} > "
            f"{max_ratio_cv}); refusing to trust time-fit-based alignment."
        )

    t_cuts = t_boundaries[t_off:t_off + n + 1]
    t_cuts_next = t_next_start[t_off:t_off + n + 1]
    r_cuts = r_boundaries[r_off:r_off + n + 1]
    r_cuts_next = r_next_start[r_off:r_off + n + 1]
    t_starts = np.concatenate(([0], t_cuts_next + 1))
    t_ends = np.concatenate((t_cuts + 1, [len(tracker)]))
    r_starts = np.concatenate(([0], r_cuts_next + 1))
    r_ends = np.concatenate((r_cuts + 1, [len(trigger_n)]))

    tracker_mask = np.zeros(len(tracker), dtype=bool)
    root_idx_parts = []
    fit_diagnostics = []
    for t_lo, t_hi, r_lo, r_hi in zip(t_starts, t_ends, r_starts, r_ends):
        if t_hi <= t_lo or r_hi <= r_lo:
            continue
        run_event_nr = tracker["run_event_nr"][t_lo:t_hi]
        root_key_local = trigger_n[r_lo:r_hi]

        fit = _fit_segment_time_calibration(run_event_nr, root_key_local,
                                            t_raw[t_lo:t_hi], r_raw[r_lo:r_hi],
                                            local_search_range, min_anchor_count, min_fit_r)
        fit_diagnostics.append((int(t_lo), int(t_hi), fit))
        if fit is None:
            continue
        slope, intercept, r_corr, n_anchor = fit

        seg_matched, seg_local_root_idx = _time_based_segment_match(
            t_raw[t_lo:t_hi], r_raw[r_lo:r_hi], slope, intercept)

        seg_indices = np.arange(t_lo, t_hi)[seg_matched]
        tracker_mask[seg_indices] = True
        root_idx_parts.append(r_lo + seg_local_root_idx[seg_matched])

    root_idx = np.concatenate(root_idx_parts) if root_idx_parts else np.empty(0, dtype=np.int64)
    match_frac = tracker_mask.sum() / len(tracker) if len(tracker) else 0.0
    if match_frac < min_match_frac:
        raise ValueError(
            f"Only {match_frac:.1%} of tracker events matched a ROOT event by time fit; "
            f"refusing to align (min_match_frac={min_match_frac:.1%})."
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


def build_aligned_tracker_branches(tracker: np.ndarray, trigger_n, root_tstamp_us, **align_kwargs):
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

    Parameters
    ----------
    **align_kwargs : passed straight through to align_tracker_to_root_by_timestamp

    Returns
    -------
    branches : dict of str -> ndarray, each shape (len(trigger_n),):
        ``tracker_x1``, ``tracker_y1``, ``tracker_x2``, ``tracker_y2``
        (sentinel-filled), plus ``tracker_matched``.
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
    return branches, match_frac
