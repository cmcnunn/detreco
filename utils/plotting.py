"""Reusable plotting and histogram helpers.

The TProfile* functions reproduce ROOT's TProfile behaviour: they bin ``x``
and, in each bin, compute the mean (and optionally the standard error) of
``y``. They replace several slow for-loop implementations scattered across
the scripts with a single vectorised version.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import mplhep as mh
from scipy.optimize import curve_fit

from utils.fit_funcs import line
from utils.constants import PITCH
from utils.data import get_run_beam


# ---------------------------------------------------------------------------
# Binned statistics
# ---------------------------------------------------------------------------
def _edges_from_bins(bins, x_min, x_max):
    if np.isscalar(bins):
        return np.linspace(x_min, x_max, int(bins) + 1)
    return np.asarray(bins)


def TProfile1d(x, y, bins, x_min=None, x_max=None, return_error=False):
    """Return per-bin mean of ``y`` as a function of ``x``.

    Vectorised equivalent of ROOT's ``TProfile``. When ``return_error`` is
    true also returns the standard error of the mean in each bin.

    Returns
    -------
    (centers, mean, counts)                       if return_error is False
    (centers, mean, error_on_mean, counts)        if return_error is True
    """
    x = np.asarray(x); y = np.asarray(y)
    if x_min is None:
        x_min = float(np.min(x))
    if x_max is None:
        x_max = float(np.max(x))

    edges = _edges_from_bins(bins, x_min, x_max)
    nbins = len(edges) - 1

    counts, _ = np.histogram(x, bins=edges)
    sum_y, _ = np.histogram(x, bins=edges, weights=y)
    sum_y2, _ = np.histogram(x, bins=edges, weights=y * y)

    nonzero = counts > 0
    mean = np.full(nbins, np.nan)
    mean[nonzero] = sum_y[nonzero] / counts[nonzero]

    centers = 0.5 * (edges[:-1] + edges[1:])

    if not return_error:
        return centers, mean, counts

    error = np.full(nbins, np.nan)
    var = np.zeros(nbins)
    var[nonzero] = sum_y2[nonzero] / counts[nonzero] - mean[nonzero] ** 2
    var = np.clip(var, 0, None)  # guard against tiny negative from FP
    error[nonzero] = np.sqrt(var[nonzero] / counts[nonzero])
    return centers, mean, error, counts


def TProfile2d(x, y, z, xbins, ybins, x_range=None, y_range=None,
               return_error=False):
    """2-D analogue of :func:`TProfile1d`: per-(x,y)-bin mean of ``z``."""
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    x_range = x_range or (float(np.min(x)), float(np.max(x)))
    y_range = y_range or (float(np.min(y)), float(np.max(y)))

    x_edges = _edges_from_bins(xbins, *x_range)
    y_edges = _edges_from_bins(ybins, *y_range)

    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    sum_z, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)
    sum_z2, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z * z)

    nonzero = counts > 0
    mean = np.full_like(counts, np.nan, dtype=float)
    mean[nonzero] = sum_z[nonzero] / counts[nonzero]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    if not return_error:
        return x_centers, y_centers, mean, counts

    var = np.zeros_like(counts, dtype=float)
    var[nonzero] = sum_z2[nonzero] / counts[nonzero] - mean[nonzero] ** 2
    var = np.clip(var, 0, None)
    error = np.full_like(counts, np.nan, dtype=float)
    error[nonzero] = np.sqrt(var[nonzero] / counts[nonzero])
    return x_centers, y_centers, mean, error, counts


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------
def get_efficiency(sel, ref, nbins, range=None):
    """Ratio of two 1-D histograms with safe divide-by-zero handling.

    Bins where the reference count is 0 are set to NaN so matplotlib draws
    them as gaps rather than zeros.
    """
    sel_hist, bin_edges = np.histogram(sel, bins=nbins, range=range)
    ref_hist, _ = np.histogram(ref, bins=nbins, range=range)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.divide(sel_hist, ref_hist,
                        out=np.zeros_like(sel_hist, dtype=float),
                        where=ref_hist > 0)
    eff[ref_hist == 0] = np.nan
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return eff, centers, sel_hist, ref_hist


# ---------------------------------------------------------------------------
# 2-D efficiency map (ratio of two 2-D histograms)
# ---------------------------------------------------------------------------
def plot_2d_efficiency_map(x_ref, y_ref, x_sel, y_sel, bins,
                           xlabel, ylabel, title, filename,
                           x_range=None, y_range=None,
                           cmin=0.0, cmax=1.0, experiment="CaloX"):
    """Build and save a 2D efficiency map.

    ``x_ref, y_ref`` are positions for reference (denominator) events;
    ``x_sel, y_sel`` are the subset that passed the selection (numerator).
    ``bins`` can be an int (number of bins each side) or an edge array.
    """
    import matplotlib.pyplot as plt
    try:
        import mplhep as mh
    except ImportError:
        mh = None

    if np.isscalar(bins):
        if x_range is None:
            x_range = (float(np.min(x_ref)), float(np.max(x_ref)))
        if y_range is None:
            y_range = (float(np.min(y_ref)), float(np.max(y_ref)))
        x_edges = np.linspace(*x_range, int(bins) + 1)
        y_edges = np.linspace(*y_range, int(bins) + 1)
    else:
        x_edges = y_edges = np.asarray(bins)

    h_ref, x_edges, y_edges = np.histogram2d(x_ref, y_ref, bins=[x_edges, y_edges])
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[x_edges, y_edges])
    eff = np.divide(h_sel, h_ref,
                    out=np.zeros_like(h_sel, dtype=float),
                    where=h_ref > 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    if mh is not None:
        pc = mh.hist2dplot(eff, x_edges, y_edges, ax=ax,
                           cmin=cmin, cmax=cmax, rasterized=True)
        if pc.cbar:
            pc.cbar.set_label("Efficiency", loc="top")
        mh.label.exp_label(exp=experiment, data=True, rlabel=title, ax=ax)
    else:
        im = ax.imshow(eff.T, origin="lower",
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       cmap="viridis", vmin=cmin, vmax=cmax)
        plt.colorbar(im, ax=ax, label="Efficiency")
        ax.set_title(title)

    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return eff, x_edges, y_edges


# ---------------------------------------------------------------------------
# 1-D profile with optional fit overlay
# ---------------------------------------------------------------------------
def plot_1d_profile_with_fit(x, y, xlabel, ylabel, title, filename,
                             bins=50, x_range=None, fit_func=None, p0=None,
                             fit_label="fit"):
    """Scatter-style TProfile plot with an optional fit overlay.

    ``fit_func`` is called as ``fit_func(x, *params)`` on the per-bin means.
    Returns the fit ``(popt, perr)`` or ``None`` if no fit was requested.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    x = np.asarray(x); y = np.asarray(y)
    centers, mean, error, counts = TProfile1d(
        x, y, bins=bins,
        x_min=None if x_range is None else x_range[0],
        x_max=None if x_range is None else x_range[1],
        return_error=True,
    )

    mask = counts > 0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(centers[mask], mean[mask], yerr=error[mask],
                fmt="o", ms=3, label="data")

    fit_result = None
    if fit_func is not None and mask.sum() >= len(p0 or ()):
        popt, pcov = curve_fit(fit_func, centers[mask], mean[mask],
                               p0=p0, sigma=error[mask] if np.all(error[mask] > 0) else None,
                               absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        xs = np.linspace(centers[mask].min(), centers[mask].max(), 400)
        ax.plot(xs, fit_func(xs, *popt), "-", label=fit_label)
        fit_result = (popt, perr)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return fit_result

def get_runtype(run_id):
    """Return a string describing the run type based on the run ID."""
    run_id = int(run_id)
    if run_id <= 1527:
        return "TB2025"
    elif 1527 < run_id <= 1630:
        return "Cosmic 2025"
    elif 1762 <= run_id <= 2043:
        return "TB2026"
    else:
        return "RUNTYPE ERROR"


def get_beam_label(run_id):
    """Return a display label combining run type with beam type/energy, e.g. ``"TB2026 pi+ 80 GeV"``.

    Falls back to just the run type when beam metadata isn't known for this
    run (e.g. TB2025, or a gap in the elog CSV's coverage).
    """
    runtype = get_runtype(run_id)
    beam_type, beam_energy_gev = get_run_beam(run_id)
    if beam_type and beam_energy_gev is not None:
        return f"{runtype} {beam_type} {beam_energy_gev} GeV"
    return runtype

def plot_profile(xh, yh, run_id, runtype="", OUTPUTDIR="/lustre/work/colnunn/detreco", label="Profile 2D", fname="profile2d"):
    if runtype == "":
        runtype = get_beam_label(run_id)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(xh, yh, bins=64)
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=label, data=True)
    ax.set_xlabel("X [cm]", loc='right')
    ax.set_ylabel("Y [cm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"{fname}_{run_id}.png"), dpi=300)
    print("Profile Plot Saved " + os.path.join(OUTPUTDIR, f"{fname}_{run_id}.png"))

def profile_mode(x, y, bins=64, min_frac=0.1, min_prominence=1.3, smooth_window=5):
    """Bin ``x`` and take the most-probable (peak) ``y`` value in each bin.

    A raw event-by-event fit gets dragged off the visible ridge by the wide,
    low-density halo of mismatched/background combinations -- that halo has
    far more points than any single fine y-bin, but the true correlation
    shows up as the *peak* of y within each x-slice, not its mean.

    Two things make a slice's "peak" untrustworthy, and both are checked:

    - Too few events: bins are only kept if their event count is at least
      ``min_frac`` of the busiest bin's count, which adapts to each run's
      statistics rather than assuming a fixed absolute cutoff. This mostly
      drops the x-range where the beam barely illuminates the detector.
    - No clear winner: a *distant*, separately-peaked y-bin rivaling the top
      bin (within ``min_prominence``) means the slice is genuinely
      ambiguous -- e.g. two real, resolvable clusters. "Distant" is found by
      walking outward from the top bin while the count keeps falling (or
      stays flat), i.e. the peak's own basin, and only searching for a
      rival outside it -- on a ``smooth_window``-moving-average of the raw
      counts, not the raw counts themselves. A real peak can be many bins
      wide (multiple scattering / strip-pitch smearing, not just one or two
      bins) and, being real data, its interior is not perfectly monotonic --
      a single-count Poisson uptick partway down its shoulder would
      otherwise look like the start of a second peak and stop the walk
      early, splitting one broad peak into a "peak" plus a phantom "rival"
      right next to it.

    If ``x`` only takes a handful of distinct values (e.g. hodoscope bar
    positions, spaced exactly one pitch apart), it's binned on those exact
    values instead of an equal-width ``linspace`` division -- the latter's
    bin width generally doesn't match the true spacing, so some bins
    straddle two positions and others fall empty, producing unevenly
    spaced profile points and sometimes a visible "kink" that isn't real.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    unique_x = np.unique(x)
    if len(unique_x) <= bins:
        x_centers = unique_x
        x_bin = np.searchsorted(unique_x, x)
        n_x_bins = len(unique_x)
    else:
        x_edges = np.linspace(x.min(), x.max(), bins + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        x_bin = np.digitize(x, x_edges) - 1
        n_x_bins = bins

    y_edges = np.linspace(y.min(), y.max(), bins + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    counts_per_bin = np.array([(x_bin == i).sum() for i in range(n_x_bins)])
    min_count = max(5, min_frac * counts_per_bin.max())

    kernel = np.ones(smooth_window) / smooth_window

    prof_x, prof_y, prof_weight = [], [], []
    for i in range(n_x_bins):
        sel = x_bin == i
        if counts_per_bin[i] < min_count:
            continue
        counts, _ = np.histogram(y[sel], bins=y_edges)
        smoothed = np.convolve(counts, kernel, mode="same")
        top = np.argmax(smoothed)

        # Walk outward from the peak while the (smoothed) count is
        # non-increasing -- that's the peak's own basin, not a competing
        # mode. Stop at the first step back uphill (a real valley), which
        # marks where a separate peak could start.
        left = top
        while left > 0 and smoothed[left - 1] <= smoothed[left]:
            left -= 1
        right = top
        while right < len(smoothed) - 1 and smoothed[right + 1] <= smoothed[right]:
            right += 1

        rival_mask = np.ones(len(smoothed), dtype=bool)
        rival_mask[left:right + 1] = False
        rival_max = smoothed[rival_mask].max() if rival_mask.any() else 0

        if rival_max > 0 and smoothed[top] < min_prominence * rival_max:
            continue

        # Report the raw-count peak within the basin, not the smoothed
        # one -- smoothing can shift "top" onto a bin with fewer (or zero)
        # raw events, which would give this profile point a misleadingly
        # low (or zero) weight in draw_fit's weighted fit.
        top_raw = left + np.argmax(counts[left:right + 1])

        prof_x.append(x_centers[i])
        prof_y.append(y_centers[top_raw])
        prof_weight.append(counts[top_raw])
    return np.array(prof_x), np.array(prof_y), np.array(prof_weight)

_FIT_OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]

def draw_fit(ax, x, y, n_sigma_clip=3.0, max_iter=5, tag=None):
    """Overlay a straight-line fit (through the per-bin peak, not raw events) and
    its equation/correlation directly on the plot.

    The fit is weighted by each profile point's own event count (a point
    built from 500 events is trusted more than one built from 6) and
    iteratively sigma-clipped: after each fit, points more than
    ``n_sigma_clip`` robust-scatter-widths (MAD-based, so a couple of bad
    points can't inflate the very spread used to judge them) from the line
    are dropped for good and the line is refit, so a handful of stray
    profile points -- e.g. from a still-misaligned segment upstream --
    can't drag the whole line off the real ridge. Rejected points are
    marked with a red X.

    ``tag`` (e.g. a run number/selection string) is prepended as the first
    line of the fit line's legend entry, if given.
    """
    prof_x, prof_y, prof_w = profile_mode(x, y)
    ax.plot(prof_x, prof_y, "o", color="white", ms=8, mec="black", mew=1)

    if len(prof_x) < 2:
        # A line fit needs at least 2 points; a low-statistics selection
        # (e.g. a tight/unvalidated cut) can leave profile_mode with 0 or 1
        # surviving bins, which would otherwise crash curve_fit.
        ax.text(0.97, 0.05, f"Not enough data for a fit ({len(prof_x)} profile point(s))",
                transform=ax.transAxes, ha="right", va="bottom",
                color="white", fontsize=20, path_effects=_FIT_OUTLINE)
        return

    keep = np.ones(len(prof_x), dtype=bool)
    m, b, cov = None, None, None
    for _ in range(max_iter):
        sigma = 1.0 / np.sqrt(prof_w[keep])
        (m, b), cov = curve_fit(line, prof_x[keep], prof_y[keep], sigma=sigma, absolute_sigma=False)
        residuals = prof_y - line(prof_x, m, b)
        mad = np.median(np.abs(residuals[keep] - np.median(residuals[keep])))
        scale = mad * 1.4826 if mad > 0 else (np.std(residuals[keep]) or 1e-9)
        outlier = keep & (np.abs(residuals) > n_sigma_clip * scale)
        if not outlier.any() or keep.sum() - outlier.sum() < 2:
            break
        keep &= ~outlier

    m_err, b_err = np.sqrt(np.diag(cov))
    r = np.corrcoef(prof_x[keep], prof_y[keep])[0, 1]
    n_rejected = int((~keep).sum())

    label = f"y = ({m:.3f} $\\pm$ {m_err:.3f})x + ({b:.3f} $\\pm$ {b_err:.3f})\n$r$ = {r:.5f}"
    if n_rejected:
        label += f"\n({n_rejected} outlier{'s' if n_rejected != 1 else ''} excluded)"
    if tag:
        label = f"{tag}\n{label}"

    xs = np.array([prof_x[keep].min(), prof_x[keep].max()])
    ax.plot(xs, line(xs, m, b), color="white", lw=2, path_effects=_FIT_OUTLINE, label=label)
    if n_rejected:
        ax.plot(prof_x[~keep], prof_y[~keep], "x", color="red", ms=12, mew=2.5)

    ax.legend(loc="lower right", fontsize=16, facecolor="black", edgecolor="white",
              labelcolor="white", framealpha=0.75)

def compute_efficiency_map(x_ref, y_ref, x_sel, y_sel, bins=64):
    """Per-bin selected/reference ratio on the standard detector-plane grid
    (+/- 32 strip pitches, matching ``plot_effhist2d``).

    Returns ``(eff, h_ref, xedges, yedges)``. ``h_ref`` is returned alongside
    ``eff`` so callers can distinguish "0% efficiency" from "no reference
    hits landed in this bin" without recomputing the histograms.
    """
    x_bins = np.linspace(-PITCH * 32, PITCH * 32, bins)
    y_bins = np.linspace(-PITCH * 32, PITCH * 32, bins)

    h_ref, xedges, yedges = np.histogram2d(x_ref, y_ref, bins=[x_bins, y_bins])
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[x_bins, y_bins])

    eff = np.divide(h_sel, h_ref, out=np.zeros_like(h_sel, dtype=float), where=h_ref > 0)
    return eff, h_ref, xedges, yedges


def intrinsic_efficiency(eff, h_ref, min_eff=0.5):
    """Mean efficiency (and binomial uncertainty) over bins with real acceptance.

    ``min_eff`` excludes bins outside the detector's geometric footprint,
    which sit at eff == 0 because they have no reference hits at all.
    """
    geometric_mask = (h_ref > 0) & (eff > min_eff)
    vals = eff[geometric_mask]
    if vals.size == 0:
        return 0.0, None
    mean = float(np.mean(vals))
    uncertainty = float(np.sqrt(mean * (1 - mean) / vals.size))
    return mean, uncertainty


def plot_effhist2d(x_ref, y_ref, x_sel, y_sel, bins, xlabel, ylabel, title, filename, runtype=""):
    eff, h_ref, xedges, yedges = compute_efficiency_map(x_ref, y_ref, x_sel, y_sel, bins)

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    pc = mh.hist2dplot(eff, xedges, yedges, ax=ax, cmin=0, cmax=1, rasterized=True)
    cb = pc.cbar
    if cb:
        cb.set_label("Efficiency", loc='top')
    mh.label.exp_label(exp="CaloX", text=runtype, data=True, rlabel=title, ax=ax)
    ax.set_xlabel(xlabel, loc='right')
    ax.set_ylabel(ylabel, loc='top')
    plt.tight_layout()
    plt.savefig(filename)
    print("Efficiency Plot Saved " + filename)
    plt.close()
    return eff, h_ref, xedges, yedges


def plot_effhist1d(x_ref, y_ref, x_sel, y_sel, title, filename):
    x_bins = np.linspace(-30, 50, 200)
    y_bins = np.linspace(-30, 50, 200)

    h_ref_x, xedges = np.histogram(x_ref, bins=x_bins)
    h_sel_x, _ = np.histogram(x_sel, bins=x_bins)
    eff_x = np.divide(h_sel_x, h_ref_x,
                      out=np.zeros_like(h_sel_x, dtype=float), where=h_ref_x > 0)

    h_ref_y, yedges = np.histogram(y_ref, bins=y_bins)
    h_sel_y, _ = np.histogram(y_sel, bins=y_bins)
    eff_y = np.divide(h_sel_y, h_ref_y,
                      out=np.zeros_like(h_sel_y, dtype=float), where=h_ref_y > 0)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    eff_x_mean = np.mean(eff_x[h_ref_x > 0])
    eff_y_mean = np.mean(eff_y[h_ref_y > 0])

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

    ax[0].step(x_centers, eff_x, where="mid")
    ax[0].set_xlabel("X [mm]")
    ax[0].set_ylabel("Efficiency")
    ax[0].set_title("Efficiency vs X")
    ax[0].text(0.02, 0.95,
               f"Mean efficiency: {eff_x_mean:.3f}\nEntries: {np.sum(h_ref_x)}",
               transform=ax[0].transAxes, verticalalignment="top",
               bbox=dict(facecolor="white", alpha=0.8))

    ax[1].step(y_centers, eff_y, where="mid")
    ax[1].set_xlabel("Y [mm]")
    ax[1].set_ylabel("Efficiency")
    ax[1].set_title("Efficiency vs Y")
    ax[1].text(0.02, 0.95,
               f"Mean efficiency: {eff_y_mean:.3f}\nEntries: {np.sum(h_ref_y)}",
               transform=ax[1].transAxes, verticalalignment="top",
               bbox=dict(facecolor="white", alpha=0.8))

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
