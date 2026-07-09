"""Generic waveform processing helpers.

All functions operate on 2D arrays of shape (n_events, n_samples) where each
row is a single event's digitized waveform. The first ``n_baseline`` samples
are used to estimate the per-event DC offset (baseline).

These helpers consolidate logic that was previously duplicated across many
analysis scripts (corrplots, dwc_testing, effplots, hgdream_track_analysis,
mcp_studies, hpcc_res_analysis, res_analysis, quick_mcp_check,
veto_waveform_plot). Keeping ``n_baseline`` as a keyword argument makes it
easy to override when a particular run has a different pre-trigger window.
"""

import numpy as np

DEFAULT_N_BASELINE = 20


def get_baselines(waveforms, n_baseline=DEFAULT_N_BASELINE):
    """Per-event mean of the first ``n_baseline`` samples.

    Returns a column vector (shape ``(n_events, 1)``) so it broadcasts against
    the original waveform array.
    """
    return np.mean(waveforms[:, :n_baseline], axis=1, keepdims=True)


def subtract_baseline(waveforms, n_baseline=DEFAULT_N_BASELINE):
    """Return ``waveforms`` with the per-event baseline subtracted."""
    return waveforms - get_baselines(waveforms, n_baseline)


def get_baseline_and_noise(waveforms, n_baseline=DEFAULT_N_BASELINE):
    """Return per-event ``(baseline, noise)``.

    ``baseline`` is the mean and ``noise`` the sample standard deviation
    (``ddof=1``) of the first ``n_baseline`` samples. Both are 1D arrays of
    length ``n_events``.
    """
    pre = waveforms[:, :n_baseline]
    baseline = np.mean(pre, axis=1)
    noise = np.std(pre, axis=1, ddof=1)
    return baseline, noise


def get_hit_times(waveforms, n_baseline=DEFAULT_N_BASELINE):
    """Sample index of the minimum of each baseline-subtracted waveform.

    Assumes hits are negative-going pulses (standard for the DWC, MCP, veto,
    and scintillator channels in this analysis).
    """
    return np.argmin(waveforms - get_baselines(waveforms, n_baseline), axis=1)


def get_saturation_mask(waveforms, threshold, n_baseline=DEFAULT_N_BASELINE):
    """Boolean mask of events that do NOT saturate past ``threshold``.

    ``True`` for good (unsaturated) events, ``False`` for saturated. Since
    ``threshold`` is negative (pulses go negative), an event is "saturated"
    when any baseline-subtracted sample falls below it.
    """
    bs = waveforms - get_baselines(waveforms, n_baseline)
    return ~np.any(bs < threshold, axis=1)
