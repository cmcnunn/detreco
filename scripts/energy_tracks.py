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
from utils.data import get_run_filepath
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

def zoomed_oscillation_window(lo, hi, period, n_periods=5):
    """Narrow [lo, hi] to n_periods oscillation cycles centered on its midpoint,
    clipped back to [lo, hi]. Falls back to the full range if period isn't a
    usable positive number (e.g. the sine fit didn't converge for that axis).
    """
    if period is None or not np.isfinite(period) or period <= 0:
        return lo, hi
    mid = 0.5 * (lo + hi)
    half_span = 0.5 * n_periods * period
    return max(lo, mid - half_span), min(hi, mid + half_span)

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
    runtype = get_beam_label(run)
    plt.style.use(mh.style.ROOT)
    fft_by_combo = {}
    for ch in ["sci", "cer"]:
        periods_by_axis = {}
        for dim in ["1d", "2d"]:
            data = energy_tracks[ch][dim]
            if dim == "1d":
                for axis in ["x", "y"]:
                    exlabel = f"{ch.upper()} Energy vs {axis.upper()} Position"
                    centers, mean, error, counts = data[axis]
                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.errorbar(centers, mean, yerr=error, fmt="-o", ms=3)
                    popt, perr, fit_mask, period, period_err = fit_sine_profile(centers, mean, error, counts=counts)
                    periods_by_axis[axis] = period
                    x_fit = np.linspace(centers[fit_mask].min(), centers[fit_mask].max(), 200)
                    y_fit = sine(x_fit, *popt)
                    fit_label = (f"period={period:.3g}±{period_err:.2g} mm")
                    ax.plot(x_fit, y_fit, "r-", linewidth=2, label=fit_label)
                    ax.set_xlabel(f"{label} {axis.upper()} Position (mm)", loc="right")
                    ax.set_ylabel(f"Average {ch} Energy (ADC)", loc="top")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    finite_mean = mean[np.isfinite(mean)]
                    if finite_mean.size:
                        lo, hi = np.nanpercentile(finite_mean, [5, 95])
                        pad = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1)
                        ax.set_ylim(lo - pad, hi + pad)
                    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True)
                    ax.grid()
                    ax.legend(fontsize=20)
                    plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}.png"))

                    plateau_lo, plateau_hi = centers[fit_mask].min(), centers[fit_mask].max()
                    zoom_lo, zoom_hi = zoomed_oscillation_window(plateau_lo, plateau_hi, period)
                    if (zoom_hi - zoom_lo) < (plateau_hi - plateau_lo):
                        ax.set_xlim(zoom_lo, zoom_hi)
                        # y autoscale doesn't recompute from the visible window on its
                        # own once set_ylim has been called above, so the oscillation
                        # amplitude (a small fraction of the full plateau's y-range)
                        # would otherwise stay squashed flat -- rescale to just the
                        # zoomed-in points instead.
                        in_zoom = (centers >= zoom_lo) & (centers <= zoom_hi) & np.isfinite(mean)
                        if np.any(in_zoom):
                            lo_y, hi_y = np.nanpercentile(mean[in_zoom], [5, 95])
                            pad_y = 0.1 * (hi_y - lo_y) if hi_y > lo_y else max(abs(lo_y), 1)
                            ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
                        plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}_zoom.png"))
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
                        mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True)
                        ax.grid()
                        ax.legend(fontsize=20)
                        plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}_filtered.png"))

                        zoom_lo, zoom_hi = zoomed_oscillation_window(x_lo, x_hi, period)
                        if (zoom_hi - zoom_lo) < (x_hi - x_lo):
                            ax.set_xlim(zoom_lo, zoom_hi)
                            in_zoom = (centers_p >= zoom_lo) & (centers_p <= zoom_hi) & np.isfinite(refined_1d)
                            if np.any(in_zoom):
                                lo_y, hi_y = np.nanpercentile(refined_1d[in_zoom], [5, 95])
                                pad_y = 0.1 * (hi_y - lo_y) if hi_y > lo_y else max(abs(lo_y), 1)
                                ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
                            plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_{axis}_filtered_zoom.png"))
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
                mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True)
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
                    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel + " (filtered)", data=True)
                    plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_filtered.png"))

                    zoom_x_lo, zoom_x_hi = zoomed_oscillation_window(x_centers[0], x_centers[-1], periods_by_axis.get("x"))
                    zoom_y_lo, zoom_y_hi = zoomed_oscillation_window(y_centers[0], y_centers[-1], periods_by_axis.get("y"))
                    if (zoom_x_hi - zoom_x_lo) < (x_centers[-1] - x_centers[0]) or (zoom_y_hi - zoom_y_lo) < (y_centers[-1] - y_centers[0]):
                        ax.set_xlim(zoom_x_lo, zoom_x_hi)
                        ax.set_ylim(zoom_y_lo, zoom_y_hi)
                        plt.savefig(os.path.join(output_dir, f"{ch}_{dim}_filtered_zoom.png"))
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
        mh.label.exp_label(ax=axes[0], exp="CaloX", text=runtype, rlabel=f"{label} Oscillation FFT", data=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "oscillation_fft.png"))
        plt.close()

    print(f"Saved energy track plots to {output_dir}")

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