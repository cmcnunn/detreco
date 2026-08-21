import os
import numpy as np
import uproot
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
from scipy.optimize import curve_fit

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
        b = 64
    else:
        b = 250

    sci_data = {
        "1d": {
            "x": TProfile1d(x_sel, sci_sel, bins=b, x_min=np.min(x_sel), x_max=np.max(x_sel), return_error=True),
            "y": TProfile1d(y_sel, sci_sel, bins=b, x_min=np.min(y_sel), x_max=np.max(y_sel), return_error=True)
        },
        "2d": TProfile2d(x_sel, y_sel, sci_sel, b, b, return_error=True),
    }
    cer_data = {
        "1d": {
            "x": TProfile1d(x_sel, cer_sel, bins=b, x_min=np.min(x_sel), x_max=np.max(x_sel), return_error=True),
            "y": TProfile1d(y_sel, cer_sel, bins=b, x_min=np.min(y_sel), x_max=np.max(y_sel), return_error=True),
        },
        "2d": TProfile2d(x_sel, y_sel, cer_sel, b, b, return_error=True),
    }
    return {"sci": sci_data, "cer": cer_data}

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
    for ch in ["sci", "cer"]:
        for dim in ["1d", "2d"]:
            data = energy_tracks[ch][dim]
            if dim == "1d":
                for axis in ["x", "y"]:
                    exlabel = f"{ch.upper()} Energy vs {axis.upper()} Position"
                    centers, mean, error, counts = data[axis]
                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.errorbar(centers, mean, yerr=error, fmt="o")
                    popt, perr, fit_mask, period, period_err = fit_sine_profile(centers, mean, error, counts=counts)
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
                    plt.close()
            elif dim == "2d":
                exlabel = f"{ch.upper()} Energy vs Position"
                x_centers, y_centers, mean, error, counts = data
                fig, ax = plt.subplots(figsize=(14, 14))
                finite_mean = mean[np.isfinite(mean)]
                vmin, vmax = np.nanpercentile(finite_mean, [40, 95]) if finite_mean.size else (None, None)
                im = ax.imshow(mean.T, origin="lower", extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]], aspect="auto", vmin=vmin, vmax=vmax)
                ax.set_xlabel(f"{label} X Position (mm)", loc="right")
                ax.set_ylabel(f"{label} Y Position (mm)", loc="top")
                mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=exlabel, data=True)
                cbar = plt.colorbar(im, ax=ax, label=f"Average {ch} Energy (ADC)")
                cbar.formatter.set_powerlimits((0, 0))
                cbar.update_ticks()
                plt.savefig(os.path.join(output_dir, f"{ch}_{dim}.png"))
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
        si_mask = (x1_raw != SENTINEL_X1) & (y1_raw != SENTINEL_Y1) & (x2_raw != SENTINEL_X2) & (y2_raw != SENTINEL_Y2)
        x1, y1 = x1_raw * 10, y1_raw * 10
        x2, y2 = x2_raw * 10, y2_raw * 10
        print(f"  [{run}] Aligned {matched.sum()}/{len(trigger_n)} ROOT events "
          f"(tracker-side match_frac={match_frac:.4%})")
    except Exception as e:
        print(f"Error processing run {run}: {e}")
        return

    try:
        hodo_energy_tracks = build_energy_tracks(hx, hy, total_sci_energy, total_cer_energy, good_hodo, calib_data=False, is_hodo=True)
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