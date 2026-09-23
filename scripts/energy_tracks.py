import os
import numpy as np
import uproot
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from utils.selectors import get_branch_names, passes_veto

from utils.selectors import get_branch_names

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:
        """No-op stand-in so the script still runs without the optional dependency."""
        def __init__(self, *a, total=None, **k):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        write = staticmethod(print)

from utils.fit_funcs import sine
from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)
from utils.hodo import reconstruct_hodoscope 
from utils.constants import VETO_THRESHOLD, X_MAPPING, Y_MAPPING
from utils.data import get_run_filepath, get_run_table_position
from utils.plotting import get_beam_label, get_runs_by_testbeam, TProfile1d, TProfile2d
from utils.energy import load_energy_data
from utils.selectors import get_branch_names

# One worker per available CPU (respects a SLURM/cgroup allocation, unlike
# os.cpu_count()); bump your job's cpu allocation and this picks it up
# automatically, no flag needed.
N_WORKERS = len(os.sched_getaffinity(0))

# Nominal si-tracker strip pitch, marked on the FFT plots for reference against
# the fitted/observed oscillation period.
EXPECTED_PITCH_MM = 4.0

# Target bin width (mm) for the silicon tracker's energy-vs-position profiles.
# The hodoscope's fixed 64 bins over its ~38mm-wide illuminated footprint
# works out to ~0.6mm/bin, which is what actually gives it its smooth-looking
# profile. Reusing a fixed *bin count* (250) for the tracker's own, typically
# much wider (~70mm), footprint spreads the same number of events over far
# more bins -- ~26 events/bin on a real run, vs ~100+/bin for the hodoscope --
# so each tracker bin's mean energy is dominated by shot noise rather than
# tracing the real modulation. Deriving the tracker's bin count from this
# fixed physical width instead keeps its per-bin statistics (and resolution
# relative to EXPECTED_PITCH_MM) comparable to the hodoscope's.
TRACKER_TARGET_BIN_MM = 0.6

def build_energy_tracks(x, y, sci, cer, sel, calib_data=True, is_hodo=True):
    """
    Build energy tracks for a given run.
    Args:
        run: Run ID to process
        x: Tracker x positions
        y: Tracker y positions
        sci: Scintillator energy data
        cer: Cherenkov energy data
        sel: Selection mask for good events
        calib_data: Whether to use calibrated energy data (default: True)
    Returns:
        matrix of 1d and 2d Profile of avg energy vs x and y positions for scintillator and Cherenkov channels.
        ["1d"]: {["x"]: centers, mean, error, counts} , {["y"]: centers, mean, error, counts}
        ["2d"]: x_centers, y_centers, mean, error, counts
    """
    # Apply selection mask
    x_sel = x[sel]
    y_sel = y[sel]
    sci_sel = sci[sel]
    cer_sel = cer[sel]

    if is_hodo:
        bx = by = 64
    else:
        bx = max(8, int(round((np.max(x_sel) - np.min(x_sel)) / TRACKER_TARGET_BIN_MM)))
        by = max(8, int(round((np.max(y_sel) - np.min(y_sel)) / TRACKER_TARGET_BIN_MM)))

    sci_data = {
        "1d": {
            "x": TProfile1d(x_sel, sci_sel, bins=bx, x_min=np.min(x_sel), x_max=np.max(x_sel), return_error=True),
            "y": TProfile1d(y_sel, sci_sel, bins=by, x_min=np.min(y_sel), x_max=np.max(y_sel), return_error=True)
        },
        "2d": TProfile2d(x_sel, y_sel, sci_sel, bx, by, return_error=True),
    }
    cer_data = {
        "1d": {
            "x": TProfile1d(x_sel, cer_sel, bins=bx, x_min=np.min(x_sel), x_max=np.max(x_sel), return_error=True),
            "y": TProfile1d(y_sel, cer_sel, bins=by, x_min=np.min(y_sel), x_max=np.max(y_sel), return_error=True),
        },
        "2d": TProfile2d(x_sel, y_sel, cer_sel, bx, by, return_error=True),
    }
    return {"sci": sci_data, "cer": cer_data}

def _nan_gaussian_filter(data, sigma):
    """Gaussian-smooth a 1D or 2D array containing NaNs, via normalized convolution.

    Ordinary gaussian_filter would treat NaN cells as contributing to their
    neighbors; instead this zero-fills them and separately smooths a validity
    mask, then divides one by the other so missing cells are excluded from the
    weighted average rather than pulling it toward zero.
    """
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    smoothed = gaussian_filter(filled, sigma=sigma, mode="nearest")
    weight = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed / weight
    result[weight < 0.5] = np.nan
    return result

def bandpass_filter(data, fine_sigma=1.0, broad_frac=0.1):
    """Isolate fine periodic structure in a 1D or 2D map: smooth lightly to
    suppress per-bin shot noise, then subtract a much broader smoothing (the
    overall beam-profile shape) so only structure finer than that broad scale
    survives.

    broad_frac is a fraction of the array's size, not a physical length -- this
    doesn't assume any particular period, it just separates "smooth large-scale
    trend" from "everything finer than that", whatever that turns out to be.
    """
    broad_sigma = max(data.shape[0] * broad_frac, fine_sigma * 4)
    fine = _nan_gaussian_filter(data, fine_sigma)
    broad = _nan_gaussian_filter(data, broad_sigma)
    return fine - broad

def _fit_sine_window(x, y, yerr, p0):
    """One curve_fit attempt (Fourier-informed p0 if not given) on an already-windowed profile."""
    if p0 is None:
        offset0 = np.mean(y)
        amp0 = (np.max(y) - np.min(y)) / 2
        dx = np.median(np.diff(x))
        # Detrend the residual plateau curvature (quadratic) before estimating the
        # frequency -- otherwise that broad, leftover hump is the strongest component
        # in the FFT and argmax locks onto it instead of the small-scale ripple.
        residual = y - np.polyval(np.polyfit(x, y, 2), x)
        spectrum = np.fft.rfft(residual)
        freqs = np.fft.rfftfreq(len(y), d=dx) * 2 * np.pi
        # Require at least 2 full cycles across the fit window -- without this, noise
        # in a short/sparse window can still make the lowest bin win argmax, producing
        # a one-giant-wavelength fit instead of the many small ones we're after.
        min_freq = 2 * (2 * np.pi) / (x.max() - x.min())
        candidates = np.where(freqs >= min_freq)[0]
        dominant = freqs[candidates[np.argmax(np.abs(spectrum[candidates]))]] if candidates.size else 1.0
        p0 = [amp0, dominant if dominant > 0 else 1.0, 0.0, offset0, 0.0]
    popt, pcov = curve_fit(sine, x, y, p0=p0, sigma=yerr, absolute_sigma=True, maxfev=10000)
    return popt, np.sqrt(np.diag(pcov))

def fit_sine_profile(centers, mean, error, counts=None, p0=None, x_range=None,
                      max_error_percentile=75, min_counts=10,
                      plateau_bands=((10, 90), (20, 80), (30, 70), (40, 60)),
                      min_points_per_cycle=3):
    """
    Fit a sine wave (utils.fit_funcs.sine) to a 1d energy profile from build_energy_tracks.

    Args:
        centers, mean, error: one axis of a build_energy_tracks 1d profile,
            e.g. energy_tracks["sci"]["1d"]["x"]
        counts: optional per-bin population from the same profile. Recommended when
            available -- TProfile1d's error has no Bessel correction, so a bin with a
            couple of entries can report a smaller error than a well-populated one and
            slip past an error-only cut. Pass it so min_counts can screen those out.
        p0: optional initial guess (amp, freq, phase, offset); skips the Fourier guess
            below entirely, for every candidate band.
        x_range: optional (lo, hi) to restrict the fit to. If given, plateau_bands is
            ignored and this exact window is used.
        max_error_percentile: drop bins whose error is above this percentile of the
            (already plateau-restricted) errors, so a few noisy points don't dominate
            the sigma-weighted fit. Set to None to disable.
        min_counts: drop bins with fewer than this many entries (only applied if
            counts is given); guards against the low-N error underestimate above.
        plateau_bands: candidate (lo_pct, hi_pct) percentile bands on mean, tried in
            turn to auto-isolate the flat-top plateau from the falling shoulders
            (only used when x_range is None). No single band generalizes: a tight
            band cleanly isolates a fine ripple on a well-populated profile but can
            alias to a spurious harmonic on a sparser one, and a wide band lets the
            (non-sinusoidal, low-frequency) shoulders back in and collapses the fit
            to one giant wavelength. Every band is fit and whichever converges with
            the smallest relative frequency uncertainty is kept.
        min_points_per_cycle: reject a candidate band if it has fewer than this many
            points per fitted oscillation cycle. A narrow band with few points can
            still report a deceptively *small* formal frequency uncertainty -- with
            barely more points than free parameters, curve_fit has little to constrain
            it against, so "smallest relative uncertainty" alone can pick a confidently
            wrong, badly-undersampled fit over a well-sampled one with an honestly
            larger error bar.
    Returns:
        popt, perr: best-fit (amp, freq, phase, offset) and their 1-sigma uncertainties.
            freq is an angular frequency in rad/mm (x is in mm), so period = 2*pi/freq.
        mask: boolean mask (into centers/mean/error) of the points actually used
        period, period_err: period in mm (2*pi/freq) and its propagated 1-sigma error
    """
    centers = np.asarray(centers); mean = np.asarray(mean); error = np.asarray(error)
    base_mask = np.isfinite(mean) & np.isfinite(error) & (error > 0)
    if counts is not None:
        base_mask &= np.asarray(counts) >= min_counts

    def _fit(mask):
        if max_error_percentile is not None:
            mask = mask & (error <= np.nanpercentile(error[mask], max_error_percentile))
        x, y, yerr = centers[mask], mean[mask], error[mask]
        popt, perr = _fit_sine_window(x, y, yerr, p0)
        period = 2 * np.pi / popt[1]
        period_err = abs(period) * (perr[1] / abs(popt[1])) if popt[1] != 0 else np.inf
        return popt, perr, mask, period, period_err

    if x_range is not None:
        mask = base_mask & (centers >= x_range[0]) & (centers <= x_range[1])
        return _fit(mask)

    best = None
    for lo_pct, hi_pct in plateau_bands:
        lo, hi = np.nanpercentile(mean[base_mask], [lo_pct, hi_pct])
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(lo), 1)
        band_mask = base_mask & (mean >= lo - pad) & (mean <= hi + pad)
        try:
            result = _fit(band_mask)
        except RuntimeError:
            continue
        popt, perr, fit_mask, period, period_err = result
        fit_x = centers[fit_mask]
        span = fit_x.max() - fit_x.min()
        points_per_cycle = fit_mask.sum() * abs(period) / span if span > 0 else 0
        if points_per_cycle < min_points_per_cycle:
            continue
        rel_freq_err = perr[1] / abs(popt[1]) if popt[1] != 0 else np.inf
        if best is None or rel_freq_err < best[0]:
            best = (rel_freq_err, result)
    if best is None:
        raise RuntimeError("fit_sine_profile: no candidate plateau band converged")
    return best[1]

def oscillation_spectrum(x, y):
    """Angular-frequency power spectrum of a quadratic-detrended, Hann-windowed
    profile -- same detrend fit_sine_profile's own p0 guess uses internally
    (see _fit_sine_window), exposed here so the fitted period can be checked
    against the dominant spectral peak instead of only trusted on curve_fit's
    say-so. x is assumed ~evenly spaced (true for a TProfile1d's bin centers).
    """
    dx = np.median(np.diff(x))
    residual = y - np.polyval(np.polyfit(x, y, 2), x)
    spectrum = np.fft.rfft(residual * np.hanning(len(residual)))
    freqs = np.fft.rfftfreq(len(residual), d=dx) * 2 * np.pi
    return freqs, np.abs(spectrum) ** 2

# Window (as a fraction of the nominal pitch) searched for a real oscillation
# peak, and the significance bar it has to clear -- see find_significant_peaks.
PEAK_SEARCH_WINDOW_FRAC = 0.3
PEAK_MIN_SIGNIFICANCE = 2.0
PEAK_LOCAL_NEIGHBORS = 4

def find_significant_peaks(centers, mean, error, counts, pitch=EXPECTED_PITCH_MM,
                            window_frac=PEAK_SEARCH_WINDOW_FRAC,
                            n_local_neighbors=PEAK_LOCAL_NEIGHBORS,
                            min_significance=PEAK_MIN_SIGNIFICANCE):
    """Find real oscillation peaks near `pitch` in this profile's FFT power
    spectrum (oscillation_spectrum), replacing the old single-sine-fit
    approach (scrapped: it picked a different period depending on how
    tightly the fit window was drawn, with no way to tell a real
    oscillation from noise -- see fit_sine_profile's history).

    Two things this gets right that a naive "tallest bin" search doesn't:

    - Significance is judged against each candidate's own *local* spectral
      neighborhood (the n_local_neighbors bins flanking the search window on
      each side), not the global spectrum. This spectrum's power falls off
      strongly with frequency in general, so comparing a candidate near the
      pitch against, say, the single largest bin anywhere (which usually
      sits at much lower frequency, where power is intrinsically higher)
      unfairly penalizes a real peak just for being at higher frequency --
      confirmed empirically: a peak that looked "insignificant" against the
      global max was still a clear, real local bump once compared to its
      own neighbors instead.
    - Every local maximum within the window that clears the significance
      bar is returned, not just the single tallest one. A profile can have
      two comparable candidate peaks close in height (confirmed on a real
      run: two bumps within ~1.2x of each other, one much closer to the
      nominal pitch than the other) -- silently picking the taller one
      would misreport which period is "the" answer when the data itself
      doesn't clearly say.

    Returns a list of {"period", "power", "significance"} dicts, sorted by
    significance descending (empty if no candidate clears the bar).
    """
    centers = np.asarray(centers); mean = np.asarray(mean)
    error = np.asarray(error); counts = np.asarray(counts)
    base_mask = np.isfinite(mean) & np.isfinite(error) & (error > 0) & (counts >= 10)
    if base_mask.sum() < 8:
        return []

    freqs, power = oscillation_spectrum(centers[base_mask], mean[base_mask])
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 2 * np.pi / np.where(freqs > 0, freqs, np.nan), np.inf)

    lo, hi = (1 - window_frac) * pitch, (1 + window_frac) * pitch
    in_window = (periods >= lo) & (periods <= hi) & (freqs > 0)
    if not in_window.any():
        return []
    win_idx = np.where(in_window)[0]

    # A local maximum here means higher power than its true spectral
    # neighbors (i-1, i+1 in the full array) -- not just the highest bin
    # among window members -- so a real bump right at the window's edge is
    # still found even though one of its neighbors falls outside the window.
    candidates = [i for i in win_idx if 0 < i < len(power) - 1
                  and power[i] >= power[i - 1] and power[i] >= power[i + 1]]
    if not candidates:
        candidates = [win_idx[np.argmax(power[win_idx])]]

    lo_bound = max(1, win_idx.min() - n_local_neighbors)
    hi_bound = min(len(freqs), win_idx.max() + 1 + n_local_neighbors)
    local_idx = np.array([i for i in range(lo_bound, hi_bound) if i not in win_idx])
    if len(local_idx) == 0:
        return []
    local_floor = np.median(power[local_idx])
    if local_floor <= 0:
        return []

    results = [{"period": float(periods[i]), "power": float(power[i]),
                "significance": float(power[i] / local_floor)}
               for i in candidates]
    results = [r for r in results if r["significance"] >= min_significance]
    results.sort(key=lambda r: -r["significance"])
    return results

# Fixed (x_lo, x_hi, y_lo, y_hi) plot bounds (mm) for each detector label,
# tight around its illuminated veto footprint. These profiles are binned out
# to the full range of veto-passing hits, but a sparse halo of scattered/
# mis-tracked hits reaches well beyond the actual beam spot at low
# per-bin counts, leaving large empty margins around the real, densely
# populated blob -- unrelated to the oscillation period, so a period-based
# zoom (the old approach) never addressed it. Measured directly off two
# TB2026 pi+ 60 GeV runs (1774, 1811) as the >=10-counts/bin pixel bounding
# box (5 counts/bin, the plots' own display threshold, still included that
# sparse halo -- 10 was the smallest threshold that consistently excluded
# it on both runs), padded by ~1.5mm and rounded; both runs agreed closely,
# so these are hardcoded rather than recomputed per run. If the beam/veto
# alignment shifts (different testbeam period, veto moved, etc.) these will
# need remeasuring the same way.
VETO_PLOT_BOUNDS = {
    "Hodoscope": {"x": (-15.0, 7.0), "y": (-20.0, 3.0)},
    "Tracker 1": {"x": (36.0, 60.0), "y": (35.0, 67.0)},
    "Tracker 2": {"x": (33.0, 56.0), "y": (35.0, 69.0)},
}

def _beam_label_with_position(run):
    """get_beam_label plus the run's table X/Y position, when known.

    Kept local to this script rather than folded into get_beam_label
    itself, since that helper is shared with scripts (eff.py, hodores.py,
    etc.) that don't plot against table position and shouldn't have their
    label cluttered by it.
    """
    label = get_beam_label(run)
    table_x, table_y = get_run_table_position(run)
    if table_x is not None and table_y is not None:
        label = f"{label} (table X={table_x:g}, Y={table_y:g} mm)"
    return label


def _rescale_y_to_window(ax, x, y, x_lo, x_hi):
    """Set ax's y-limits from the 5th/95th percentile of y restricted to
    [x_lo, x_hi] -- autoscale doesn't recompute from the visible window on
    its own once set_xlim narrows it to less than the full data range.
    """
    in_window = (x >= x_lo) & (x <= x_hi) & np.isfinite(y)
    if not np.any(in_window):
        return
    lo_y, hi_y = np.nanpercentile(y[in_window], [5, 95])
    pad_y = 0.1 * (hi_y - lo_y) if hi_y > lo_y else max(abs(lo_y), 1)
    ax.set_ylim(lo_y - pad_y, hi_y + pad_y)

def plot_energy_tracks(run, energy_tracks, label="Hodoscope"):
    '''
    Plot the energy tracks for a given run.
    Args:
        run: Run ID to process
        energy_tracks: Dictionary containing energy track data for scintillator and Cherenkov channels
        label: Label for the plot (default: "Hodoscope")
    Will save all plots to /output/energy_tracks/{run}/{label}
    '''
    output_dir = os.path.join("output", "energy_tracks", run, label.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    runtype = _beam_label_with_position(run)
    plt.style.use(mh.style.ROOT)
    bounds = VETO_PLOT_BOUNDS.get(label)
    fft_by_combo = {}
    for ch in ["sci", "cer"]:
        for dim in ["1d", "2d"]:
            data = energy_tracks[ch][dim]
            if dim == "1d":
                for axis in ["x", "y"]:
                    exlabel = f"{ch.upper()} Energy vs {axis.upper()} Position"
                    centers, mean, error, counts = data[axis]
                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.errorbar(centers, mean, yerr=error, fmt="-o", ms=3)
                    popt, perr, fit_mask, period, period_err = fit_sine_profile(centers, mean, error, counts=counts)
                    x_fit = np.linspace(centers[fit_mask].min(), centers[fit_mask].max(), 200)
                    y_fit = sine(x_fit, *popt)
                    fit_label = (f"period={period:.3g}±{period_err:.2g} mm")
                    ax.plot(x_fit, y_fit, "r-", linewidth=2, label=fit_label)
                    ax.set_xlabel(f"{label} {axis.upper()} Position (mm)", loc="right")
                    ax.set_ylabel(f"Average {ch} Energy (ADC)", loc="top")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    if bounds is not None:
                        x_lo, x_hi = bounds[axis]
                        ax.set_xlim(x_lo, x_hi)
                        _rescale_y_to_window(ax, centers, mean, x_lo, x_hi)
                    else:
                        finite_mean = mean[np.isfinite(mean)]
                        if finite_mean.size:
                            lo, hi = np.nanpercentile(finite_mean, [5, 95])
                            pad = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1)
                            ax.set_ylim(lo - pad, hi + pad)
                    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True, fontsize=24)
                    ax.grid()
                    ax.legend(fontsize=20)
                    plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}.png"))
                    plt.close()

                    freqs, power = oscillation_spectrum(centers[fit_mask], mean[fit_mask])
                    fft_by_combo[(ch, axis)] = (freqs, power, period)

                    # Same band-pass idea as the 2D filtered map, but on the 1D profile
                    # directly: no sine model imposed, just light smoothing minus a
                    # broad smoothing, so any periodic residual is visible on its own
                    # merits rather than only through a fitted sine curve.
                    # Restricted to the same plateau window the sine fit already
                    # validated -- outside it, statistics fall and the profile bends
                    # sharply (the beam edge), and the broad-smoothing term reacts to
                    # that real transition with large spurious swings that look like
                    # oscillation but are a boundary artifact, not signal.
                    x_lo, x_hi = centers[fit_mask].min(), centers[fit_mask].max()
                    plateau = (centers >= x_lo) & (centers <= x_hi)
                    centers_p = centers[plateau]
                    mean_masked_1d = np.where(counts[plateau] >= 5, mean[plateau], np.nan)
                    refined_1d = bandpass_filter(mean_masked_1d)
                    finite_refined_1d = refined_1d[np.isfinite(refined_1d)]
                    if finite_refined_1d.size:
                        fig, ax = plt.subplots(figsize=(12, 12))
                        ax.plot(centers_p, refined_1d, "o-", ms=3, label="filtered data")
                        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
                        # Overlay just the oscillatory term of the sine fit (offset and
                        # the linear m*x term zeroed out) -- that background is exactly
                        # what the filter above already subtracted, so this is the
                        # model's prediction for what's left, on the same footing as
                        # the model-free filtered data.
                        x_fit_p = np.linspace(x_lo, x_hi, 200)
                        y_fit_p = sine(x_fit_p, popt[0], popt[1], popt[2], 0, 0)
                        ax.plot(x_fit_p, y_fit_p, "r-", linewidth=2, label=fit_label)
                        ax.set_xlabel(f"{label} {axis.upper()} Position (mm)", loc="right")
                        ax.set_ylabel(f"{ch} Energy, background-subtracted (ADC)", loc="top")
                        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                        if bounds is not None:
                            vx_lo, vx_hi = bounds[axis]
                            ax.set_xlim(vx_lo, vx_hi)
                            _rescale_y_to_window(ax, centers_p, refined_1d, vx_lo, vx_hi)
                        mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True, fontsize=24)
                        ax.grid()
                        ax.legend(fontsize=20)
                        plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}_filtered.png"))
                        plt.close()


            elif dim == "2d":
                exlabel = f"{ch.upper()} Energy vs Position"
                x_centers, y_centers, mean, error, counts = data
                min_2d_counts = 5
                mean_masked = np.where(counts >= min_2d_counts, mean, np.nan)
                fig, ax = plt.subplots(figsize=(14, 14))
                finite_mean = mean_masked[np.isfinite(mean_masked)]
                im = ax.imshow(mean_masked.T, origin="lower", extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]], aspect="auto")
                ax.set_xlabel(f"{label} X Position (mm)", loc="right")
                ax.set_ylabel(f"{label} Y Position (mm)", loc="top")
                cbar = plt.colorbar(im, ax=ax, label=f"Average {ch} Energy (ADC)")
                cbar.formatter.set_powerlimits((0, 0))
                cbar.update_ticks()
                mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True, fontsize=24)
                if bounds is not None:
                    ax.set_xlim(*bounds["x"])
                    ax.set_ylim(*bounds["y"])
                plt.savefig(os.path.join(output_dir, f"{ch}_{dim}.png"))
                plt.close()

                # Band-pass: light smoothing to kill per-pixel shot noise, minus a much
                # broader smoothing (the overall beam-profile shape) to remove that
                # trend -- whatever fine structure is left over is shown on its own,
                # without assuming what scale it should be at.
                refined = bandpass_filter(mean_masked)
                finite_refined = refined[np.isfinite(refined)]
                if finite_refined.size:
                    vabs = np.nanpercentile(np.abs(finite_refined), 95)
                    fig, ax = plt.subplots(figsize=(14, 14))
                    im = ax.imshow(refined.T, origin="lower",
                                    extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                                    aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
                    ax.set_xlabel(f"{label} X Position (mm)", loc="right")
                    ax.set_ylabel(f"{label} Y Position (mm)", loc="top")
                    cbar = plt.colorbar(im, ax=ax, label=f"{ch} Energy, background-subtracted (ADC)")
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel + " (filtered)", data=True, fontsize=24)
                    if bounds is not None:
                        ax.set_xlim(*bounds["x"])
                        ax.set_ylim(*bounds["y"])
                    plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_filtered.png"))
                    plt.close()

    combos = [(ch, axis) for ch in ["sci", "cer"] for axis in ["x", "y"] if (ch, axis) in fft_by_combo]
    if combos:
        fig, axes = plt.subplots(len(combos), 1, figsize=(12, 5 * len(combos)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (ch, axis) in zip(axes, combos):
            freqs, power, period = fft_by_combo[(ch, axis)]
            ax.plot(freqs[1:], power[1:], "-o", ms=3)
            ax.axvline(2 * np.pi / period, color="r", ls="--", label=f"sine fit: {period:.3g} mm")
            ax.axvline(2 * np.pi / EXPECTED_PITCH_MM, color="g", ls=":", label=f"{EXPECTED_PITCH_MM:.3g} mm pitch")
            ax.set_ylabel(f"{ch.upper()} vs {axis.upper()}\nPower", loc="top")
            ax.grid()
            ax.legend(fontsize=14)
        axes[-1].set_xlabel("Angular frequency (rad/mm)", loc="right")
        mh.label.exp_label(ax=axes[0], exp="CaloX", text=runtype, rlabel=f"{label} Oscillation FFT", data=True, fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "oscillation_fft.png"))
        plt.close()

    print(f"Saved energy track plots to {output_dir}")

def plot_significant_peaks(run, energy_tracks, label="Hodoscope"):
    """Plot each axis's FFT power spectrum (SCI and CER overlaid) with every
    peak find_significant_peaks judges real marked and labeled, alongside
    the search window and local-comparison band it was judged against.

    Separate from plot_energy_tracks (which still does its own, sine-fit-
    based per-axis plots) rather than replacing anything there -- this is
    the new, validated peak-search approach, additive so the existing
    plots keep working exactly as before regardless of how this one turns
    out. Saved as oscillation_peaks.png alongside plot_energy_tracks's own
    output for the same run/label.
    """
    output_dir = os.path.join("output", "energy_tracks", run, label.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    runtype = _beam_label_with_position(run)
    plt.style.use(mh.style.ROOT)

    lo, hi = (1 - PEAK_SEARCH_WINDOW_FRAC) * EXPECTED_PITCH_MM, (1 + PEAK_SEARCH_WINDOW_FRAC) * EXPECTED_PITCH_MM
    colors = {"sci": "tab:blue", "cer": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, axis in zip(axes, ["x", "y"]):
        any_data = False
        for ch in ["sci", "cer"]:
            centers, mean, error, counts = energy_tracks[ch]["1d"][axis]
            centers = np.asarray(centers); mean = np.asarray(mean)
            error = np.asarray(error); counts = np.asarray(counts)
            base_mask = np.isfinite(mean) & np.isfinite(error) & (error > 0) & (counts >= 10)
            if base_mask.sum() < 8:
                continue
            any_data = True
            freqs, power = oscillation_spectrum(centers[base_mask], mean[base_mask])
            ax.plot(freqs[1:], power[1:], "-", lw=1.5, color=colors[ch], alpha=0.8, label=ch.upper())
            for p in find_significant_peaks(centers, mean, error, counts):
                pf = 2 * np.pi / p["period"]
                ax.plot(pf, p["power"], "o", color=colors[ch], ms=10, mec="black")
                ax.annotate(f"{p['period']:.2f} mm ({p['significance']:.1f}x)", (pf, p["power"]),
                            textcoords="offset points", xytext=(8, 8), fontsize=11,
                            color=colors[ch], fontweight="bold")
        if not any_data:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.axvspan(2 * np.pi / hi, 2 * np.pi / lo, color="grey", alpha=0.12,
                   label=f"search window (±{PEAK_SEARCH_WINDOW_FRAC:.0%} of pitch)")
        ax.axvline(2 * np.pi / EXPECTED_PITCH_MM, color="green", ls=":", lw=2,
                   label=f"{EXPECTED_PITCH_MM:.0f} mm pitch")
        ax.set_yscale("log")
        ax.set_xlabel("Angular frequency (rad/mm)", loc="right")
        ax.set_ylabel("Power", loc="top")
        ax.legend(fontsize=11, loc="upper right")
        mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=f"{label} {axis.upper()}", data=True, fontsize=24)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "oscillation_peaks.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Peak-search plot saved to {outpath}")

def process_run(run):
    """
    Build and save all energy track plots for a single run.
    Args:
        run: Run ID to process (str or int)
    """
    run = str(run)
    try:
        with uproot.open(get_run_filepath(int(run))) as f:
            t = f["EventTree"]
            # Load Hodoscope Energy
            veto_branch, _, _ = get_branch_names(run)
            HGx = np.stack(t["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
            HGy = np.stack(t["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
            veto_wf = np.stack(t[veto_branch].array(library="np"))
            trigger_n = t["trigger_n"].array(library="np")
            root_tstamp = t["FERS_Board1_tstamp_us"].array(library="np")

        veto_sel = passes_veto(veto_wf, threshold=VETO_THRESHOLD)
        hx, hy, good_hodo = reconstruct_hodoscope(HGx, HGy)
        #Load Tracker data
        si_data = load_tracker_run(int(run))
        #Load energy data
        total_sci_energy, total_cer_energy = load_energy_data(int(run), calib_data=False)

        tracker_branches, match_frac = build_aligned_tracker_branches(si_data, trigger_n, root_tstamp)
        x1_raw, y1_raw = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
        x2_raw, y2_raw = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]
        matched = tracker_branches["tracker_matched"]
        # Sentinel mask must run on the raw values -- SENTINEL_X1 etc are in raw units,
        # so comparing against them after the /10 mm conversion below silently passes
        # every no-hit row through instead of filtering it out.
        si_mask = (x1_raw != SENTINEL_X1) & (y1_raw != SENTINEL_Y1) & (x2_raw != SENTINEL_X2) & (y2_raw != SENTINEL_Y2) & veto_sel
        x1, y1 = x1_raw * 10, y1_raw * 10
        x2, y2 = x2_raw * 10, y2_raw * 10
        print(f"  [{run}] Aligned {matched.sum()}/{len(trigger_n)} ROOT events "
          f"(tracker-side match_frac={match_frac:.4%})")
    except Exception as e:
        print(f"Error processing run {run}: {e}")
        return

    try:
        hodo_energy_tracks = build_energy_tracks(hx, hy, total_sci_energy, total_cer_energy, good_hodo & veto_sel, calib_data=False, is_hodo=True)
        trk1_energy_tracks = build_energy_tracks(x1, y1, total_sci_energy, total_cer_energy, si_mask, calib_data=False, is_hodo=False)
        trk2_energy_tracks = build_energy_tracks(x2, y2, total_sci_energy, total_cer_energy, si_mask, calib_data=False, is_hodo=False)
        plot_energy_tracks(run, hodo_energy_tracks, label="Hodoscope")
        plot_energy_tracks(run, trk1_energy_tracks, label="Tracker 1")
        plot_energy_tracks(run, trk2_energy_tracks, label="Tracker 2")
        plot_significant_peaks(run, hodo_energy_tracks, label="Hodoscope")
        plot_significant_peaks(run, trk1_energy_tracks, label="Tracker 1")
        plot_significant_peaks(run, trk2_energy_tracks, label="Tracker 2")
    except Exception as e:
        print(f"Error plotting run {run}: {e}")

def main():
    parser = argparse.ArgumentParser(
    description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    parser.add_argument("--testbeam", type=str,
                         help="Process every run belonging to this testbeam/run period "
                              "(e.g. TB2025, TB2026, 'Cosmic 2025'), instead of a single --run")
    args = parser.parse_args()

    if args.testbeam:
        runs = get_runs_by_testbeam(args.testbeam)
        if not runs:
            print(f"No runs found for testbeam '{args.testbeam}'")
            return
        print(f"Processing {len(runs)} runs from {args.testbeam} using {N_WORKERS} workers")
        with tqdm(total=len(runs), desc=args.testbeam, unit="run") as pbar, \
            ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(process_run, run): run for run in runs}
            for fut in as_completed(futures):
                run = futures[fut]
                pbar.set_postfix_str(f"run {run}")
                try:
                    fut.result()
                except Exception as e:
                    tqdm.write(f"Error processing run {run}: {e}")
                pbar.update(1)
    elif args.run:
        process_run(args.run)
    else:
        parser.error("one of --run or --testbeam is required")

if __name__ == "__main__":
    main()