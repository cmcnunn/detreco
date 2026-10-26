"""Robust 1D residual statistics, shared by the hodoscope-vs-track studies
(``scripts/hodores.py``'s resolution, ``utils/si_efficiency.py``'s
tracker-referenced hodoscope efficiency).

Both take residuals that are a real, flat-topped peak sitting on a broad
tail of tracker-to-ROOT mismatches -- see ``scripts.hodores``'s module
docstring for why the clip and the truncation are needed.
"""

import numpy as np

MAD_CLIP = 5.0  # sigma-equivalent clip: calibration fit sample, and truncated_rms's starting window
RMS_TRUNCATION_SIGMA = 3.0  # truncated_rms keeps events within this many RMS of the mean
MIN_RMS_EVENTS = 20

# (half-range, z) pairs for robust_clip_mask's percentile-widening fallback:
# z is the standard-normal quantile with P(-z < Z < z) = (hi - lo) / 100,
# so spread / (2*z) estimates a Gaussian sigma from that pair the same way
# IQR / 1.349 does for the 25/75 pair (1.349 == 2 * 0.6745, included here
# as the first, tightest rung of the same ladder).
_ROBUST_PERCENTILES = ((25, 75, 0.6745), (10, 90, 1.2816), (5, 95, 1.6449), (1, 99, 2.3263))


def robust_clip_mask(values, n_mad=MAD_CLIP):
    """Boolean mask marking which points to keep, robust to a dominant spike value.

    Guards against the broad mismatch tail a partially-wrong tracker
    alignment produces (see ``scripts.hodores``'s module docstring), the
    same MAD-based approach ``utils.plotting.draw_fit`` uses for its line
    fit -- with a fallback for when MAD itself collapses to 0.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        scale = mad * 1.4826
        return np.abs(values - median) <= n_mad * scale

    # MAD collapses to exactly 0 whenever at least half the sample sits on
    # one exact value -- e.g. a single dominant hodoscope bar. A single
    # fixed fallback percentile pair isn't safe here either: confirmed on
    # real data (run 1774, si1/hodoY, in a narrow 0.3mm tracker-position
    # slice) that the dominant bar alone can hold *more* than 50% of the
    # sample, which pins the 25th/75th percentiles to that same exact value
    # too (IQR == 0), so a plain IQR fallback would clip away the genuinely
    # real, immediately-adjacent bars along with the actual outlier tail --
    # 1205 events collapsed to the 673 sitting exactly on the mode,
    # discarding 217 + 198 events one bar to either side that are real
    # hodoscope smearing, not mismatches. Instead, widen the percentile pair
    # step by step until one finds a nonzero spread; if even the 1st/99th
    # percentiles coincide (essentially everything on one value), there's
    # nothing meaningful left to clip against, so keep everything and let
    # the downstream estimate (and its own truncation) handle it instead.
    for lo, hi, z in _ROBUST_PERCENTILES:
        q_lo, q_hi = np.percentile(values, [lo, hi])
        spread = q_hi - q_lo
        if spread > 0:
            scale = spread / (2 * z)
            return np.abs(values - median) <= n_mad * scale
    return np.ones(len(values), dtype=bool)


def truncated_rms(values, n_sigma=RMS_TRUNCATION_SIGMA, max_iter=100):
    """RMS of ``values`` within +/- ``n_sigma`` RMS of their mean, iterated until stable.

    Starts from ``robust_clip_mask``'s window, so the iteration begins on
    the real peak rather than the broad mismatch background (see
    ``scripts.hodores``'s module docstring), then repeatedly recomputes the
    mean and RMS of the kept events and keeps everything within ``n_sigma``
    RMS of that mean, until the kept set stops changing.

    The uncertainty uses the kept sample's own fourth moment,
    var(RMS) ~= (m4 - RMS^4) / (4 RMS^2 N), rather than the Gaussian
    RMS / sqrt(2N): the residual is flat-topped, so the two differ. It is
    statistical only -- the truncation itself is a choice, not a fluctuation.

    Returns ``(mean, rms, rms_err, n_used, n_total, keep)``, or ``None`` if
    fewer than MIN_RMS_EVENTS survive.
    """
    values = np.asarray(values)
    n_total = len(values)
    keep = robust_clip_mask(values)
    for _ in range(max_iter):
        if keep.sum() < MIN_RMS_EVENTS:
            return None
        mean, rms = values[keep].mean(), values[keep].std()
        new_keep = np.abs(values - mean) <= n_sigma * rms
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    kept = values[keep]
    if len(kept) < MIN_RMS_EVENTS:
        return None
    mean, rms = kept.mean(), kept.std()
    if rms <= 0:
        return None
    m4 = np.mean((kept - mean) ** 4)
    rms_err = np.sqrt(max(m4 - rms ** 4, 0.0) / (4 * rms ** 2 * len(kept)))
    return mean, rms, rms_err, len(kept), n_total, keep
