"""Event selection masks.

Each function returns a 1D boolean array, ``True`` where the event passes.
They compose naturally: ``mask = passes_veto(v) & mcp_hit_mask(m) & ...``.
"""

import numpy as np

from .constants import VETO_THRESHOLD
from .plotting import get_runtype
from .waveforms import DEFAULT_N_BASELINE, get_baselines, get_saturation_mask


# MCP amplitude cut. Scripts historically used -40 ADC after baseline subtraction.
DEFAULT_MCP_THRESHOLD = -40.0

# Last FERS-phase run of TB2026; MCP1/MCP2 move to different DRS channels
# once the DRS readout takes over (runs 1821+).
TB2026_FERS_PHASE_LAST_RUN = 1820

# The 1cm/3cm test counters were only added to the DAQ starting this run;
# reading their branches for an earlier run would KeyError on a branch that
# doesn't exist yet, so gate on this explicitly instead.
COUNTER_1CM_3CM_FIRST_RUN = 1825


def get_branch_names(run_id):
    """Return the ``(veto, mcp1, mcp2)`` branch names for ``run_id``.

    These branch names changed between test-beam periods, and within
    TB2026 the MCP1/MCP2 channels moved when the DRS readout replaced the
    FERS-phase digitizer (run 1821). The veto branch is unchanged across
    TB2026.
    """
    runtype = get_runtype(run_id)
    run_id = int(run_id)

    if runtype == "TB2025":
        return (
            "DRS_Board7_Group1_Channel6",
            "DRS_Board0_Group3_Channel6",
            "DRS_Board0_Group3_Channel7",
        )

    if runtype == "TB2026":
        veto = "DRS_Brg1_Board0_Group0_Channel7"
        if run_id <= TB2026_FERS_PHASE_LAST_RUN:
            return veto, "DRS_Brg1_Board0_Group0_Channel5", "DRS_Brg1_Board0_Group0_Channel6"
        return veto, "DRS_Brg1_Board3_Group3_Channel6", "DRS_Brg1_Board3_Group3_Channel7"

    raise ValueError(f"No veto/MCP branch mapping for run {run_id} (runtype={runtype!r})")


def get_mcp_pulse_window_ns(run_id):
    """Return the ``(start, end)`` ns window in which the MCP pulse falls for ``run_id``.

    The MCP pulse arrival time shifts with the readout electronics: TB2025's
    DRS setup, TB2026's FERS-phase digitizer, and TB2026's later DRS phase
    (run 1821+) each introduce a different cable/digitizer delay.
    """
    runtype = get_runtype(run_id)
    run_id = int(run_id)

    if runtype == "TB2025":
        return (100.0, 130.0)

    if runtype == "TB2026":
        if run_id <= TB2026_FERS_PHASE_LAST_RUN:
            return (34.0, 44.0)
        return (84.0, 100.0)

    raise ValueError(f"No MCP pulse window mapping for run {run_id} (runtype={runtype!r})")


def get_counter_branch_names(run_id):
    """Return the ``(counter_1cm, counter_3cm)`` branch names for ``run_id``.

    These two small test counters were only added to the DAQ starting run
    ``COUNTER_1CM_3CM_FIRST_RUN``; raises for any earlier run.
    """
    run_id = int(run_id)
    if run_id < COUNTER_1CM_3CM_FIRST_RUN:
        raise ValueError(
            f"No 1cm/3cm counter branches for run {run_id}: only present "
            f"from run {COUNTER_1CM_3CM_FIRST_RUN} onward."
        )
    return "DRS_Brg1_Board0_Group1_Channel0", "DRS_Brg1_Board0_Group1_Channel1"


def passes_veto(veto_wf, threshold=VETO_THRESHOLD, n_baseline=DEFAULT_N_BASELINE):
    """True for events whose veto channel did NOT dip below ``threshold``.

    The veto plane fires (goes very negative) when a particle passes through
    a region we want to exclude; we keep events whose baseline-subtracted
    minimum stays *above* that threshold.
    """
    bs_min = (veto_wf - get_baselines(veto_wf, n_baseline)).min(axis=1)
    return bs_min > threshold


def mcp_hit_mask(mcp_wf, amplitude_threshold=DEFAULT_MCP_THRESHOLD,
                 shoulder_threshold=None, require_interior_peak=True,
                 n_baseline=DEFAULT_N_BASELINE):
    """True for events where the MCP channel shows a real pulse.

    Criteria (all applied, following the logic in mcp_studies.py):
      - baseline-subtracted minimum below ``amplitude_threshold``
      - if ``shoulder_threshold`` is set, the two samples on either side of the
        minimum must also be below it (rejects single-sample spikes)
      - if ``require_interior_peak``, the minimum must NOT be at index 0 or
        at the last sample (rejects pulses clipped at the edges)
    """
    bs = mcp_wf - get_baselines(mcp_wf, n_baseline)
    min_idx = np.argmin(bs, axis=1)
    n_samples = bs.shape[1]

    mask = bs.min(axis=1) < amplitude_threshold

    if shoulder_threshold is not None:
        rows = np.arange(len(bs))
        idx_plus = np.clip(min_idx + 1, 0, n_samples - 1)
        idx_minus = np.clip(min_idx - 1, 0, n_samples - 1)
        mask &= (bs[rows, idx_plus] < shoulder_threshold)
        mask &= (bs[rows, idx_minus] < shoulder_threshold)

    if require_interior_peak:
        mask &= (min_idx > 0) & (min_idx < n_samples - 1)

    return mask


# Both counters show the same shape: a narrow noise pedestal (baseline-
# subtracted minimum, Gaussian, centered ~-12 to -14 ADC counts) that's
# fully gone by about -20/-21, separated from a broad continuum of real
# pulses reaching out to full-scale saturation (~-3950). -25 sits with
# clear margin past the noise edge while keeping the smallest real pulses.
# Derived from baseline-subtracted-minimum histograms of both channels
# across 5 runs spanning 1827-2042 (~115k events); no gap/valley is fully
# empty, so this isn't a hard boundary, just a well-clear-of-noise cut.
DEFAULT_COUNTER_THRESHOLD = -25.0


def counter_1cm_hit_mask(wf, amplitude_threshold=DEFAULT_COUNTER_THRESHOLD, **kwargs):
    """True for events where the 1 cm test counter shows a real pulse.

    Same pulse-finding criteria as ``mcp_hit_mask`` (see there for
    ``shoulder_threshold``/``require_interior_peak``/``n_baseline``).
    """
    return mcp_hit_mask(wf, amplitude_threshold=amplitude_threshold, **kwargs)


def counter_3cm_hit_mask(wf, amplitude_threshold=DEFAULT_COUNTER_THRESHOLD, **kwargs):
    """True for events where the 3 cm test counter shows a real pulse.

    Same pulse-finding criteria as ``mcp_hit_mask`` (see there for
    ``shoulder_threshold``/``require_interior_peak``/``n_baseline``).
    """
    return mcp_hit_mask(wf, amplitude_threshold=amplitude_threshold, **kwargs)


