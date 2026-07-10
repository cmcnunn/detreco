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
