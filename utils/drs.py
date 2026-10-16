"""TB2026 calorimeter energy from the DRS waveforms, by CaloXDataAnalysis's method.

CaloXDataAnalysis (variables/drs.py, and compute_cfd_integral in utils/functions.cc)
turns each DRS channel's 1024-sample waveform into one energy per event:

1. subtract the baseline: the upper median of samples [0, 200);
2. flip the 6mm channels -- every TB2026 DRS channel is amplified, and the amplified
   6mm ones read out inverted (get_drs_branches_to_flip) -- so every pulse is positive;
3. take the highest sample of the whole record as the peak; below MIN_AMPLITUDE the
   energy is 0;
4. walk back from the peak to the last sample at or below CFD_FRACTION of it, and sum
   the samples from WINDOW_PRE before the next one to WINDOW_POST after it (50 samples,
   10 ns at 0.2 ns/sample).

There is no DRS gain calibration in CaloXDataAnalysis, so energies are summed ADC counts.
Each DRS channel is matched to a FERS channel by data/drs/DRS_PHASE_{1,2}.csv (copied
from CaloXDataAnalysis data/channel_maps/; the DRS was recabled at run 1896), which also
gives its fibre type: Scintillation -> sci, Quartz and Plastic -> cer.

Two things seen on run 1866 that this method, being CaloX's, does not correct: every
channel has narrow 1-3-sample spikes in ~10-20% of events, and on a channel with no
pulse the peak search settles on noise or baseline drift, so nearly all 220 channels
add a positive sum every event (~40% of the event total).
"""

import csv
import os

import numpy as np
import uproot

from .data import get_run_filepath
from .io import ensure_output_dir, project_root

DRS_MAP_DIR = os.path.join(project_root(), "data", "drs")
# CaloXDataAnalysis channels/channel_map.py: _DRS_FULL_RUN_TB2026 (first run whose DRS
# map comes from the CSV) and _DRS_PHASE2_RUN (recabling).
DRS_FIRST_RUN = 1828
DRS_PHASE2_FIRST_RUN = 1896
# TB2026's 3mm FERS boards; DRS channels matched to any other board are 6mm.
FERS_3MM_BOARDS = ("6", "7")

BASELINE_SAMPLES = 200
MIN_AMPLITUDE = 5.0
CFD_FRACTION = 0.2
WINDOW_PRE, WINDOW_POST = 5, 45

# Events per pulse_energy call when integrating a channel, to bound memory.
_CHUNK = 5000


def drs_channel_map(run):
    """[(branch, kind, flip)] for the run's DRS calorimeter channels.

    kind is "sci" or "cer"; flip is True for the 6mm channels (see module docstring).
    """
    run = int(run)
    if run < DRS_FIRST_RUN:
        raise ValueError(f"run {run} predates the TB2026 DRS channel map (first run {DRS_FIRST_RUN})")
    phase = 2 if run >= DRS_PHASE2_FIRST_RUN else 1
    with open(os.path.join(DRS_MAP_DIR, f"DRS_PHASE_{phase}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [(row["DRS_ROOT_Bridge_Mapping"].replace("DRS_Bridge", "DRS_Brg"),
             "sci" if row["type"] == "Scintillation" else "cer",
             row["fers_root_board"] not in FERS_3MM_BOARDS)
            for row in rows]


def pulse_energy(waveforms, flip=False):
    """CaloXDataAnalysis's per-event DRS energy for one channel, shape (n_events,)."""
    s = np.asarray(waveforms, dtype=np.float32)
    mid = BASELINE_SAMPLES // 2
    s = s - np.partition(s[:, :BASELINE_SAMPLES], mid, axis=1)[:, mid, None]
    if flip:
        s = -s
    n, length = s.shape
    rows = np.arange(n)
    peak = s.argmax(axis=1)
    amp = s[rows, peak]

    idx = np.arange(length, dtype=np.int32)
    below = (s <= CFD_FRACTION * amp[:, None]) & (idx <= peak[:, None])
    crossing = np.where(below, idx, 0).max(axis=1)  # walk back stops at sample 0

    cumsum = np.zeros((n, length + 1))
    np.cumsum(s, axis=1, out=cumsum[:, 1:])
    start = np.clip(crossing + 1 - WINDOW_PRE, 0, length)
    end = np.clip(crossing + 1 + WINDOW_POST, 0, length)
    return np.where(amp >= MIN_AMPLITUDE, cumsum[rows, end] - cumsum[rows, start], 0.0)


def drs_channel_energies(run, use_cache=True):
    """(energies, channel_map): energies is (n_events, n_channels) float32, one column
    per drs_channel_map entry.

    Reading and integrating all 220 channels takes ~5 min per run, so the result is
    cached in output/drs_energy/<run>.npz and reused while its channel list matches.
    """
    channel_map = drs_channel_map(run)
    branches = [branch for branch, _, _ in channel_map]
    cache_path = os.path.join(ensure_output_dir("drs_energy"), f"{int(run)}.npz")
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if list(cached["branches"]) == branches:
            return cached["energies"], channel_map

    # array_cache=None: uproot's default "100 MB" array cache sizes a jagged branch
    # read with library="np" by its object array of pointers, not the waveforms, so it
    # kept every channel read -- ~125 MB per channel, ~28 GB per worker for a 30k-event
    # run, which took a 32-worker --testbeam pass to OUT_OF_MEMORY even at 224 GB.
    with uproot.open(get_run_filepath(int(run)), array_cache=None) as f:
        tree = f["EventTree"]
        energies = np.empty((tree.num_entries, len(channel_map)), dtype=np.float32)
        for k, (branch, _, flip) in enumerate(channel_map):
            waveforms = np.stack(tree[branch].array(library="np"))
            for lo in range(0, len(waveforms), _CHUNK):
                energies[lo:lo + _CHUNK, k] = pulse_energy(waveforms[lo:lo + _CHUNK], flip)
    # Write then rename, so a job killed mid-write can't leave a truncated cache behind.
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp_path, "wb") as fh:
        np.savez(fh, energies=energies, branches=np.array(branches))
    os.replace(tmp_path, cache_path)
    return energies, channel_map


def load_drs_energy(run, use_cache=True):
    """Per-event (total_sci, total_cer) DRS energy in summed ADC counts -- the DRS
    counterpart of utils.energy.load_energy_data."""
    energies, channel_map = drs_channel_energies(run, use_cache=use_cache)
    is_sci = np.array([kind == "sci" for _, kind, _ in channel_map])
    return energies[:, is_sci].sum(axis=1, dtype=np.float64), energies[:, ~is_sci].sum(axis=1, dtype=np.float64)
