"""Beam particle ID from the TB2026 service-DRS counters.

The same selection CaloXDataAnalysis uses (``configs/selection_config.py``,
``core/selection_manager.py``), so both repos agree on what counts as an
electron or a muon event:

    muon     : TTUMuonVeto fired
    electron : TTUMuonVeto not fired, Cer474 and Cer519 fired
    pion     : TTUMuonVeto not fired, Cer519 fired, Cer474 not
    proton   : TTUMuonVeto not fired, neither Cherenkov fired

Events matching none of these (Cer474 fired without Cer519) are left
unclassified, as CaloXDataAnalysis's beam-composition check does.

How far to trust each class (checked 2026-10-22 against the calorimeter SCI
sum on runs 1812-1817, 1768, 1770, 1809):

- The muon tag is clean. Muon-tagged events deposit a MIP-like SCI sum in
  every run checked, including inside e+ and pi+ beams. It also shows the
  nominal e+ 120-160 GeV beams are 75-90% muons, and pi+ 160 GeV ~23%.
- The Cherenkov split depends on how the counters' pressures were set for
  each run. In the e+ 10-80 GeV runs 12-17% of events land in "pion" while
  depositing full electron energy: Cer474 inefficiency, not real pions. So
  "electron" there undercounts, and ``~muon`` is the robust split.
"""

import numpy as np
import uproot

from .data import get_run_filepath
from .plotting import get_runtype

# Service-DRS counters used for PID in TB2026: branch, pulse window
# [ts_min, ts_max) in DRS samples, and the summed-amplitude cut that counts as
# "fired" -- CaloXDataAnalysis's _SERVICE_DRS_CUTS / _SERVICE_DRS_CUTS_TB2026.
# (HoleVeto, the hodoscope veto, is Channel7 on the same group; see
# selectors.get_branch_names.)
PID_COUNTERS_TB2026 = {
    "TTUMuonVeto": ("DRS_Brg1_Board0_Group0_Channel1", 200, 400, 5e3),
    "Cer474": ("DRS_Brg1_Board0_Group0_Channel2", 400, 800, 2000.0),
    "Cer519": ("DRS_Brg1_Board0_Group0_Channel3", 400, 800, 1000.0),
}

# The cuts above were tuned on CaloXDataAnalysis's baseline -- the median of
# the first 200 samples -- not waveforms.get_baselines' mean of the first 20,
# so use the same here.
PID_BASELINE_SAMPLES = 200

# Which counters must fire (True) or stay quiet (False) for each particle.
PARTICLE_REQUIREMENTS = {
    "muon": {"TTUMuonVeto": True},
    "electron": {"TTUMuonVeto": False, "Cer474": True, "Cer519": True},
    "pion": {"TTUMuonVeto": False, "Cer474": False, "Cer519": True},
    "proton": {"TTUMuonVeto": False, "Cer474": False, "Cer519": False},
}

# Which classes count as each nominal beam's own particle, for picking it out
# of a mixed beam. pi+ takes protons too: both are hadrons to a tracker, and
# the pion/proton split hangs on the Cherenkov pressures (pi+ 160 GeV tags 71%
# "proton"). Electrons Cer474 missed land in "pion", so selecting per nominal
# beam drops them from e+ instead of mixing them into another beam's hadrons.
BEAM_PARTICLES = {"e+": ("electron",), "pi+": ("pion", "proton"), "mu+": ("muon",)}

# Waveforms are read this many events at a time: a full run's three 1024-sample
# channels at once is ~1 GB.
DEFAULT_STEP_SIZE = 10000


def get_pid_counters(run_id):
    """Return ``{counter: (branch, ts_min, ts_max, cut)}`` for ``run_id``.

    Only TB2026 is mapped. TB2025 used different channels and a PSD (see
    CaloXDataAnalysis's PARTICLE_SELECTIONS_TB2025), not ported here.
    """
    runtype = get_runtype(run_id)
    if runtype == "TB2026":
        return dict(PID_COUNTERS_TB2026)
    raise ValueError(f"No PID counter mapping for run {run_id} (runtype={runtype!r})")


def counter_fired(waveforms, ts_min, ts_max, cut, n_baseline=PID_BASELINE_SAMPLES):
    """True for events whose counter pulse, summed over ``[ts_min, ts_max)``,
    exceeds ``cut``.

    The pulses go negative, so the baseline-subtracted waveform is flipped
    before summing -- as CaloXDataAnalysis does for every TB2026 service
    channel on bridge 1.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    baseline = np.median(waveforms[:, :n_baseline], axis=1, keepdims=True)
    return (baseline - waveforms)[:, ts_min:ts_max].sum(axis=1) > cut


def particle_masks(fired):
    """``{particle: mask}`` from ``{counter: fired mask}`` (see PARTICLE_REQUIREMENTS)."""
    masks = {}
    for particle, requirements in PARTICLE_REQUIREMENTS.items():
        mask = np.ones(len(next(iter(fired.values()))), dtype=bool)
        for counter, should_fire in requirements.items():
            mask &= fired[counter] if should_fire else ~fired[counter]
        masks[particle] = mask
    return masks


def load_counter_fired(run_id, step_size=DEFAULT_STEP_SIZE):
    """``{counter: fired mask}`` over every ROOT event of ``run_id``, in the
    same event order as the other per-event readers (files in run_list.json
    order). Reads the waveforms ``step_size`` events at a time."""
    counters = get_pid_counters(run_id)
    filepaths = get_run_filepath(run_id)
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    branches = [branch for branch, _, _, _ in counters.values()]
    chunks = {counter: [] for counter in counters}
    for arrs in uproot.iterate([f"{fp}:EventTree" for fp in filepaths], branches,
                               step_size=step_size, library="np"):
        for counter, (branch, ts_min, ts_max, cut) in counters.items():
            chunks[counter].append(counter_fired(np.stack(arrs[branch]), ts_min, ts_max, cut))
    return {counter: np.concatenate(parts) for counter, parts in chunks.items()}


def load_particle_masks(run_id, step_size=DEFAULT_STEP_SIZE):
    """``{particle: mask}`` over every ROOT event of ``run_id`` -- e.g.
    ``load_particle_masks(run)["electron"] & passes_veto(...)``."""
    return particle_masks(load_counter_fired(run_id, step_size=step_size))


def load_beam_particle_mask(run_id, beam_type, step_size=DEFAULT_STEP_SIZE):
    """Events of ``run_id`` tagged as the nominal ``beam_type``'s own particle
    (see BEAM_PARTICLES), over every ROOT event."""
    if beam_type not in BEAM_PARTICLES:
        raise ValueError(f"No PID selection for beam type {beam_type!r} (have {sorted(BEAM_PARTICLES)})")
    masks = load_particle_masks(run_id, step_size=step_size)
    return np.logical_or.reduce([masks[p] for p in BEAM_PARTICLES[beam_type]])
