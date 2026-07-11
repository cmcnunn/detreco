"""Loader for the raw silicon-tracker ASCII dumps (``TrackerData/by_run/<run>``).

One ``.dat`` file per spill, one line per event, 29 whitespace-separated
fields. Station 3 is permanently unplugged (its y-column is a constant
-6000 sentinel), so only stations 1 and 2 are exposed here. The last 11
fields are DAQ bookkeeping used to match a tracker event to its calorimeter
(DREAM) event; see field notes below.
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
_COL_RUN_EVENT_NR = 25       # hex; "event from start of run + 1"
_COL_DREAM_EVENT_NR = 26     # hex; DREAM's own event counter
_COL_GLOBAL_EVENT_NR = 28    # decimal; same value as _COL_RUN_EVENT_NR

TRACKER_DTYPE = np.dtype([
    ("x1", "f8"), ("y1", "f8"),
    ("x2", "f8"), ("y2", "f8"),
    ("event_in_spill", "i8"),
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
                 _COL_EVENT_IN_SPILL, _COL_RUN_EVENT_NR,
                 _COL_DREAM_EVENT_NR, _COL_GLOBAL_EVENT_NR),
        converters={
            _COL_EVENT_IN_SPILL: hex16,
            _COL_RUN_EVENT_NR: hex16,
            _COL_DREAM_EVENT_NR: hex16,
        },
        ndmin=2,
    )
    out = np.empty(raw.shape[0], dtype=TRACKER_DTYPE)
    out["x1"], out["y1"], out["x2"], out["y2"] = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    out["event_in_spill"] = raw[:, 4]
    out["run_event_nr"] = raw[:, 5]
    out["dream_event_nr"] = raw[:, 6]
    out["global_event_nr"] = raw[:, 7]
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
