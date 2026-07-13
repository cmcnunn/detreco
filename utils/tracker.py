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
``event_flag`` for run 1771 -- all constant or all-zero placeholders), so
``align_tracker_to_root``/``align_tracker_to_root_by_timestamp`` remain
inference, not exact lookups.
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


def align_tracker_to_root(tracker: np.ndarray, root_key, search_range: int = 50,
                          min_match_frac: float = 0.95):
    """Match tracker events to a DREAM ROOT ntuple's event counter branch.

    The tracker's ``run_event_nr`` counts continuously from the start of the
    tracker's own DAQ run (which spans many physics runs), so it's related to
    DREAM's per-physics-run event counter by a run-specific constant offset:

        tracker["run_event_nr"] - offset == root_key

    The offset is found by a brute-force search near the naive guess (the
    difference of the two arrays' minimums), picking whichever offset within
    +/- ``search_range`` matches the most events.

    IMPORTANT: pass ``root_key = tree["trigger_n"].array(...)``, not
    ``event_n``. ``event_n`` is a perfectly dense 0..N-1 range with zero
    gaps, so *any* offset that keeps the shifted values in-bounds "matches"
    100% of the time regardless of whether it's the right offset -- it can't
    discriminate at all. ``trigger_n`` has a handful of real gaps, which
    genuinely constrains the search, but on run 1771 there were only 3 gaps
    across 113k events, giving a match-rate curve that's a broad, nearly
    flat plateau (99.97%-99.99%) across roughly +/-30 events rather than a
    sharp peak. In other words: this pins the offset down to within maybe a
    few dozen events, good enough for bulk/statistical comparisons (e.g.
    correlating tracker vs. hodoscope hit positions across many events), but
    NOT reliable for exact event-by-event matching. For that, use the
    tracker's timestamp fields (not currently exposed by this loader)
    against DREAM's ``FERS_Board*_tstamp_us`` branches instead -- real
    timestamps are far less ambiguous than these near-dense counters.

    Very short/aborted runs (e.g. run 1864) can fall well short of even this
    loose match rate, because the tracker's own spill boundaries don't land
    on the physics-run boundary in that case -- ``min_match_frac`` guards
    against silently trusting a bad offset for runs like that.

    Returns
    -------
    tracker_mask : ndarray of bool, shape (len(tracker),)
        True for tracker events that found a matching ROOT event.
    root_idx : ndarray of int
        Row index into ``root_key`` for each True entry of ``tracker_mask``,
        in the same order -- so ``root_array[root_idx]`` lines up with
        ``tracker[tracker_mask]``.
    offset : int
    match_frac : float
    """
    root_key = np.asarray(root_key)
    naive_offset = int(tracker["run_event_nr"].min() - root_key.min())

    best_offset, best_n = naive_offset, -1
    for delta in range(-search_range, search_range + 1):
        offset = naive_offset + delta
        n = np.isin(tracker["run_event_nr"] - offset, root_key).sum()
        if n > best_n:
            best_offset, best_n = offset, n

    match_frac = best_n / len(tracker) if len(tracker) else 0.0
    if match_frac < min_match_frac:
        raise ValueError(
            f"Only {match_frac:.1%} of tracker events matched a ROOT event "
            f"at the best offset found ({best_offset}); refusing to align "
            f"(min_match_frac={min_match_frac:.1%}). This run's tracker spill "
            f"boundaries likely don't line up with the physics run boundary "
            f"(seen e.g. on very short runs)."
        )

    shifted = tracker["run_event_nr"] - best_offset
    key_lookup = {}
    for i, v in enumerate(root_key):
        key_lookup.setdefault(v, i)
    tracker_mask = np.isin(shifted, root_key)
    root_idx = np.fromiter(
        (key_lookup[v] for v in shifted[tracker_mask]),
        dtype=np.int64, count=int(tracker_mask.sum()),
    )
    return tracker_mask, root_idx, best_offset, match_frac


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
    """Row indices ``i`` where ``counter[i+1] - counter[i]`` marks a spill boundary.

    The beam turns off between spills, so any monotonic per-event clock
    (the tracker's raw hardware counter, or a ROOT timestamp branch) shows a
    gap there orders of magnitude larger than the typical inter-event
    spacing -- ``threshold_factor`` is applied to the *median* diff so this
    adapts to either clock's tick rate without needing to know it.
    Occasionally the same physical gap splits across two adjacent diffs
    (e.g. one extra event lands right at the boundary); ``merge_gap`` merges
    candidates within that many rows of each other into a single boundary.
    ``counter`` is unwrapped first (see ``_unwrap_counter``) in case an
    overflow happens to land right at a spill boundary -- otherwise that
    boundary's real jump would show up as a huge negative diff instead and
    be missed entirely, silently merging two spills into one segment.
    """
    counter = _unwrap_counter(counter, modulus)
    if len(counter) < 2:
        return np.empty(0, dtype=np.int64)
    diffs = np.diff(counter)
    median = np.median(diffs)
    if median <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.where(diffs > median * threshold_factor)[0]
    if len(candidates) == 0:
        return candidates.astype(np.int64)
    merged = [candidates[0]]
    for c in candidates[1:]:
        if c - merged[-1] <= merge_gap:
            merged[-1] = c
        else:
            merged.append(c)
    return np.array(merged, dtype=np.int64)


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


def align_tracker_to_root_by_timestamp(tracker: np.ndarray, trigger_n, root_tstamp_us,
                                       threshold_factor: float = 300, merge_gap: int = 3,
                                       max_shift: int = 10, min_segments: int = 5,
                                       max_ratio_cv: float = 0.1, local_search_range: int = 200,
                                       min_segment_match_frac: float = 0.5,
                                       min_match_frac: float = 0.5):
    """Match tracker events to a DREAM ROOT ntuple using real hardware timestamps.

    ``align_tracker_to_root`` finds *one* constant offset for the whole run
    by brute-force search against ``trigger_n`` -- but that counter is
    nearly gapless, so (per its docstring) the search can only pin the
    offset down to within a few dozen events, not exactly, and -- discovered
    empirically while building this function -- the true offset isn't even
    constant across the whole run to begin with. Individual dropped/extra
    rows on either side (not just at spill boundaries) mean the true
    ``run_event_nr - offset == trigger_n`` relationship drifts by a handful
    of events over the course of a run; a single global offset, however
    precisely found, cannot fit more than one piece of it at a time.

    This function instead finds spill boundaries -- a far more distinctive,
    independent signal built from the timestamp the tracker itself sent to
    DREAM (``tracker["timestamp_to_dream"]``, from previously-unused DAQ
    columns) and DREAM's ``FERS_Board*_tstamp_us`` branch (a 32-bit counter
    double-packed into a 64-bit float -- take the low 32 bits) -- matches
    them up between the two streams (see ``find_spill_boundaries`` /
    ``_match_boundary_offset``), and then re-derives the offset
    *independently within each spill segment* via the same brute-force
    search as ``align_tracker_to_root``, restricted to that segment's own
    small slice of ``trigger_n``. Restricting the search this way is what
    makes it precise: matching against a few thousand local values is far
    more discriminating than matching against the whole run's near-gapless
    counter, so the search reliably lands on the exact per-segment offset
    rather than a compromise across drift it can't see.

    Parameters
    ----------
    tracker : structured array from ``load_tracker_run`` (needs ``timestamp``)
    trigger_n : DREAM's ``trigger_n`` branch
    root_tstamp_us : any one board's ``FERS_Board*_tstamp_us`` branch, same
        length/order as ``trigger_n``
    local_search_range : int
        Offset search radius around each segment's own naive guess.
    min_segment_match_frac : float
        A segment whose best local offset still matches fewer than this
        fraction of its own rows is dropped rather than trusted.

    Returns
    -------
    tracker_mask : ndarray of bool, shape (len(tracker),)
    root_idx : ndarray of int, one entry per True in ``tracker_mask``
    offsets : list of int, the offset used for each spill segment (for
        diagnostics -- there is no single global offset by design)
    match_frac : float, fraction of all tracker events successfully matched
    """
    trigger_n = np.asarray(trigger_n)
    root_ts = (np.asarray(root_tstamp_us).astype(np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.int64)

    t_boundaries = find_spill_boundaries(tracker["timestamp_to_dream"], threshold_factor, merge_gap,
                                        modulus=2 ** 31)
    r_boundaries = find_spill_boundaries(root_ts, threshold_factor, merge_gap,
                                        modulus=2 ** 32)

    match = _match_boundary_offset(t_boundaries, r_boundaries, max_shift, min_segments)
    if match is None:
        raise ValueError(
            f"Not enough spill boundaries to align by timestamp "
            f"(tracker: {len(t_boundaries)}, root: {len(r_boundaries)}; need >= {min_segments} segments)."
        )
    t_off, r_off, n, score = match
    if score > max_ratio_cv:
        raise ValueError(
            f"Best spill-boundary match is still inconsistent (cv={score:.3f} > "
            f"{max_ratio_cv}); refusing to trust timestamp-based alignment."
        )

    t_cuts = t_boundaries[t_off:t_off + n + 1]
    r_cuts = r_boundaries[r_off:r_off + n + 1]
    t_starts = np.concatenate(([0], t_cuts + 1))
    t_ends = np.concatenate((t_cuts + 1, [len(tracker)]))
    r_starts = np.concatenate(([0], r_cuts + 1))
    r_ends = np.concatenate((r_cuts + 1, [len(trigger_n)]))

    key_lookup = {}
    for i, v in enumerate(trigger_n):
        key_lookup.setdefault(v, i)

    tracker_mask = np.zeros(len(tracker), dtype=bool)
    root_idx_parts = []
    offsets = []
    for t_lo, t_hi, r_lo, r_hi in zip(t_starts, t_ends, r_starts, r_ends):
        seg = tracker[t_lo:t_hi]
        root_key_local = trigger_n[r_lo:r_hi]
        if len(seg) == 0 or len(root_key_local) == 0:
            continue

        # NOTE: a segment's own local trigger_n slice is frequently
        # completely gapless (the whole run only has a handful of real
        # gaps total, per align_tracker_to_root's docstring), so more than
        # one nearby offset -- typically an exact off-by-one twin -- can
        # tie for the same match count while only one is actually right;
        # no amount of retrying this same count-based check can tell them
        # apart. Untested tie-break ideas (continuity with the previous
        # segment's offset) made results *worse* by propagating an early
        # wrong guess forward; this still needs a real fix (likely
        # resolving the exact drop point within a segment via row-by-row
        # sequence alignment, rather than one offset per whole segment).
        naive_offset = int(seg["run_event_nr"].min() - root_key_local.min())
        best_offset, best_n = naive_offset, -1
        for delta in range(-local_search_range, local_search_range + 1):
            offset = naive_offset + delta
            cnt = np.isin(seg["run_event_nr"] - offset, root_key_local).sum()
            if cnt > best_n:
                best_offset, best_n = offset, cnt
        if best_n / len(seg) < min_segment_match_frac:
            continue
        offsets.append(best_offset)

        shifted = seg["run_event_nr"] - best_offset
        seg_mask = np.isin(shifted, trigger_n)
        seg_indices = np.arange(t_lo, t_hi)[seg_mask]
        tracker_mask[seg_indices] = True
        root_idx_parts.append(np.fromiter(
            (key_lookup[v] for v in shifted[seg_mask]),
            dtype=np.int64, count=int(seg_mask.sum()),
        ))

    root_idx = np.concatenate(root_idx_parts) if root_idx_parts else np.empty(0, dtype=np.int64)
    match_frac = tracker_mask.sum() / len(tracker) if len(tracker) else 0.0
    if match_frac < min_match_frac:
        raise ValueError(
            f"Only {match_frac:.1%} of tracker events matched a ROOT event "
            f"across {len(offsets)} usable segments; refusing to align "
            f"(min_match_frac={min_match_frac:.1%})."
        )
    return tracker_mask, root_idx, offsets, match_frac
