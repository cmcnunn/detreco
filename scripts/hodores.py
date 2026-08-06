"""Hodoscope spatial resolution against the silicon tracker, two ways.

Both methods below compare the hodoscope's reconstructed position to a
si-tracker station's, using the same reference selection (hodoscope-good,
veto-passing, tracker-hit events, matched to ROOT via
utils.tracker.build_aligned_tracker_branches -- same as sihodocor.py/
sieff.py). Both measure not the hodoscope's resolution in isolation, but
that resolution convolved (in quadrature) with the tracker station's own
resolution and multiple scattering across the gap between them:

    sigma_measured^2 ~= sigma_hodo^2 + sigma_tracker^2 + sigma_MS^2

Since the si tracker's point resolution is expected to be far finer than
the hodoscope's 0.6mm bar pitch, sigma_tracker^2 should be negligible next
to the other two terms -- both methods below lean on that assumption
rather than measuring sigma_tracker independently and subtracting it. (The
rigorous alternative -- extrapolating a track through both tracker
stations to the hodoscope's own z and subtracting that extrapolation's
known uncertainty in quadrature, as in e.g. beam-telescope DUT resolution
analyses -- needs station/hodoscope z-positions this module doesn't have;
worth revisiting if/when those are available.)

Method 1 -- full-sample residual fit (``fit_resolution``): histograms
hodo-minus-tracker over every reference-selected event and fits a Gaussian.
Fast, uses all statistics, but trusts the sigma_tracker ~ 0 assumption
without checking it.

Method 2 -- window convergence (``window_convergence``): slices events into
progressively narrower bins of tracker position and fits the hodoscope's
*raw* position within each slice. As the window narrows well below the
tracker's own resolution, both the tracker's contribution and the window's
own geometric width (variance width^2/12 for a uniform window) shrink
toward zero, so the fitted sigma should fall and then plateau at the same
target the quadrature approach assumes: sigma_hodo^2 + sigma_MS^2. Slower
(most events get cut away at small widths) but doesn't just assume
sigma_tracker is negligible -- it demonstrates it, and a plateau that
hasn't been reached by the smallest usable window is a visible warning
sign the two methods' results should be compared to.

The tracker-to-ROOT alignment is itself imperfect (see utils/tracker.py's
module docstring and scan_alignment_correlation): a run can report a high
match_frac while carrying little genuine per-event correspondence, which
would show up here as a residual distribution with no clean central peak
rather than a real (if broad) Gaussian core. Both methods guard against
this with a MAD-based clip (same approach as utils.plotting.draw_fit) so a
minority of mismatched events can't drag the fit off the real peak, but a
low retained-fraction is a sign the alignment for that run/station is
untrustworthy, not a bug in this script. Method 1 clips its own (large,
stable) full-sample residual directly. Method 2 clips *once*, globally, on
that same full-sample residual before ever slicing into windows -- not
per-window -- because a percentile-based clip threshold recomputed fresh on
each window's own small, heavily bar-quantized subsample turns out to be
unstable (see ``window_convergence``'s docstring for the specific failure
mode this caused on real data).

The hodoscope and tracker each define their own (0, 0), and those origins
are not surveyed to coincide -- confirmed on real data: residuals sit near
a constant few-cm offset (not zero) for every station/axis. Both fits
below center their window on the data's own (robust) median rather than an
assumed absolute range for exactly this reason -- sigma is the target, the
fitted mean is just a by-product.

Usage:
    python -m scripts.hodores --run <run_id>   # one run
    python -m scripts.hodores                  # every run in run_list.json, pooled
"""

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot
from scipy.optimize import curve_fit

from utils.constants import HG_THRESHOLD, PITCH, X_MAPPING, Y_MAPPING
from utils.data import get_run_filepath, load_run_list
from utils.fit_funcs import gauss
from utils.hodo import reconstruct_hodoscope
from utils.io import ensure_output_dir
from utils.plotting import get_beam_label
from utils.selectors import get_branch_names, passes_veto
from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)

OUTPUT_DIR = ensure_output_dir("hodores")

DEFAULT_HALF_WIDTH_MM = 5.0  # residual-fit window is [median - this, median + this]
DEFAULT_BINS = 100
MAD_CLIP = 5.0  # sigma-equivalent clip applied before every fit below
MAX_RELATIVE_SIGMA_ERR = 0.5  # reject a fit if sigma_err exceeds this fraction of sigma

# Descending tracker-window widths (mm) for the convergence scan. 0.6mm is
# the hodoscope's own bar pitch, included as a natural reference point.
DEFAULT_WIDTHS_MM = [8.0, 4.0, 2.0, 1.0, 0.6, 0.3, 0.15, 0.08, 0.01]
DEFAULT_MIN_WINDOW_EVENTS = 50

LABELS = {"x1": "Station 1 X", "y1": "Station 1 Y", "x2": "Station 2 X", "y2": "Station 2 Y"}


def load_run_data(run_id):
    """Return ``(data, match_frac)`` for one run.

    ``data`` holds, per station/axis key ("x1", "y1", "x2", "y2"), a
    ``(tracker_mm, hodo_mm)`` pair of same-length arrays: tracker position
    (mm) and the hodoscope's reconstructed position (mm) for the
    reference-selected events (good hodoscope hit, veto pass) that also
    registered a real hit on that station -- i.e. already restricted to
    real tracker hits, no sentinel rows.

    Raises on any failure (missing ROOT/tracker file, bad alignment, etc.)
    so callers can decide how to skip a run.
    """
    filepath = get_run_filepath(run_id)
    veto_branch, _, _ = get_branch_names(run_id)
    with uproot.open(filepath) as f:
        tree = f["EventTree"]
        trigger_n = tree["trigger_n"].array(library="np")
        root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
        hg_x = np.stack(tree["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
        hg_y = np.stack(tree["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
        veto_wf = np.stack(tree[veto_branch].array(library="np"))

    xh, yh, good_hodo = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH)
    veto_sel = passes_veto(veto_wf)

    si_data = load_tracker_run(run_id)
    tracker_branches, match_frac = build_aligned_tracker_branches(si_data, trigger_n, root_tstamp)
    x1, y1 = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
    x2, y2 = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]

    ref = good_hodo & veto_sel
    hit1 = ref & (x1 != SENTINEL_X1) & (y1 != SENTINEL_Y1)
    hit2 = ref & (x2 != SENTINEL_X2) & (y2 != SENTINEL_Y2)

    # tracker x1/y1/x2/y2 are in cm; *10 puts them in mm alongside xh/yh.
    data = {
        "x1": (10 * x1[hit1], xh[hit1]), "y1": (10 * y1[hit1], yh[hit1]),
        "x2": (10 * x2[hit2], xh[hit2]), "y2": (10 * y2[hit2], yh[hit2]),
    }
    return data, match_frac


# (half-range, z) pairs for _robust_clip's percentile-widening fallback:
# z is the standard-normal quantile with P(-z < Z < z) = (hi - lo) / 100,
# so spread / (2*z) estimates a Gaussian sigma from that pair the same way
# IQR / 1.349 does for the 25/75 pair (1.349 == 2 * 0.6745, included here
# as the first, tightest rung of the same ladder).
_ROBUST_PERCENTILES = ((25, 75, 0.6745), (10, 90, 1.2816), (5, 95, 1.6449), (1, 99, 2.3263))


def _robust_clip_mask(values, n_mad=MAD_CLIP):
    """Boolean mask marking which points to keep, robust to a dominant spike value.

    Guards against the broad mismatch tail a partially-wrong tracker
    alignment produces (see module docstring), the same MAD-based approach
    ``utils.plotting.draw_fit`` uses for its line fit -- with a fallback for
    when MAD itself collapses to 0.
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
    # real data (run 1774, si1/hodoY at a 0.3mm window) that the dominant
    # bar alone can hold *more* than 50% of the sample, which pins the
    # 25th/75th percentiles to that same exact value too (IQR == 0), so a
    # plain IQR fallback would clip away the genuinely real,
    # immediately-adjacent bars along with the actual outlier tail -- 1205
    # events collapsed to the 673 sitting exactly on the mode, discarding
    # 217 + 198 events one bar to either side that are real hodoscope
    # smearing, not mismatches. Instead, widen the percentile pair step by
    # step until one finds a nonzero spread; if even the 1st/99th
    # percentiles coincide (essentially everything on one value), there's
    # nothing meaningful left to clip against, so keep everything and let
    # the downstream fit's own validity gate (MAX_RELATIVE_SIGMA_ERR) catch
    # a genuinely bad fit instead.
    for lo, hi, z in _ROBUST_PERCENTILES:
        q_lo, q_hi = np.percentile(values, [lo, hi])
        spread = q_hi - q_lo
        if spread > 0:
            scale = spread / (2 * z)
            return np.abs(values - median) <= n_mad * scale
    return np.ones(len(values), dtype=bool)


def _robust_clip(values, n_mad=MAD_CLIP):
    """Drop points far from the median before fitting. See ``_robust_clip_mask``."""
    values = np.asarray(values)
    return values[_robust_clip_mask(values, n_mad=n_mad)]


def _fit_gaussian(values, bins=DEFAULT_BINS, half_width=DEFAULT_HALF_WIDTH_MM):
    """Histogram + Gaussian fit on already-clean ``values`` -- no outlier clipping.

    Low-level piece shared by ``fit_resolution`` (which clips first, using
    its own local sample) and ``window_convergence`` (which clips once,
    globally, before slicing into windows -- see that function's docstring
    for why re-clipping per window is unsafe). The histogram window is
    centered on the data's own median rather than a fixed absolute range --
    e.g. the hodoscope and tracker frames carry their own, uncoordinated
    origins, so a residual's true peak generally sits well away from zero
    (see module docstring).

    A successful ``curve_fit`` call doesn't guarantee a meaningful result --
    a histogram with too few well-separated bins to actually constrain a
    Gaussian (e.g. a couple of dominant spike bins plus mostly-empty
    neighbors) can converge to a numerically "valid" but physically
    meaningless degenerate solution (huge sigma, huge sigma_err), rather
    than raising. ``MAX_RELATIVE_SIGMA_ERR`` catches that after the fact:
    a sigma whose own uncertainty is comparable to (or bigger than) its
    value isn't a measurement, it's noise that happened to fit.

    Returns ``(mu, mu_err, sigma, sigma_err, n_used, (centers, counts, popt))``,
    or ``None`` if there isn't enough data to fit, or the fit isn't trustworthy.
    """
    values = np.asarray(values)
    n_used = len(values)
    if n_used < 20:
        return None

    center = np.median(values)
    hist_range = (center - half_width, center + half_width)
    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nonzero = counts > 0
    if nonzero.sum() < 4:
        return None

    p0 = (counts.max(), np.median(values), np.std(values) or 1.0)
    try:
        popt, pcov = curve_fit(gauss, centers[nonzero], counts[nonzero], p0=p0,
                               sigma=np.sqrt(counts[nonzero]), absolute_sigma=True, maxfev=10000)
    except RuntimeError:
        return None

    perr = np.sqrt(np.diag(pcov))
    mu, sigma = popt[1], abs(popt[2])
    mu_err, sigma_err = perr[1], perr[2]
    if not np.isfinite(sigma_err) or sigma <= 0 or sigma_err > MAX_RELATIVE_SIGMA_ERR * sigma:
        return None
    return mu, mu_err, sigma, sigma_err, n_used, (centers, counts, popt)


def fit_resolution(values, bins=DEFAULT_BINS, half_width=DEFAULT_HALF_WIDTH_MM):
    """Clip outliers from one 1-D distribution (a residual, typically), then fit a Gaussian.

    Thin wrapper around ``_fit_gaussian``: clips first using this sample's
    own statistics. Fine for a single big, stable sample (Method 1's
    full-run residual) -- see ``window_convergence`` for why a per-window
    version of this same clip-then-fit pattern is unsafe on small samples.

    Returns ``(mu, mu_err, sigma, sigma_err, n_used, n_total, (centers, counts, popt))``,
    or ``None`` if there isn't enough data to fit, or the fit isn't trustworthy.
    """
    values = np.asarray(values)
    n_total = len(values)
    clipped = _robust_clip(values)
    result = _fit_gaussian(clipped, bins=bins, half_width=half_width)
    if result is None:
        return None
    mu, mu_err, sigma, sigma_err, n_used, extra = result
    return mu, mu_err, sigma, sigma_err, n_used, n_total, extra


def _diagnose_fit_failure(values, bins=DEFAULT_BINS, half_width=DEFAULT_HALF_WIDTH_MM):
    """Best-effort human-readable reason a ``_fit_gaussian`` call on ``values`` returned ``None``.

    ``values`` is expected to already be outlier-clipped (as
    ``window_convergence`` passes it) -- this does not clip again, it just
    re-derives why the histogram/fit step failed, purely for a readable
    message in place of a bare ``--`` in the convergence table.
    """
    values = np.asarray(values)
    if len(values) < 20:
        return "too few events"
    center = np.median(values)
    counts, _ = np.histogram(values, bins=bins, range=(center - half_width, center + half_width))
    n_distinct_bins = int((counts > 0).sum())
    if n_distinct_bins < 4:
        # The hodoscope only reports discrete, PITCH-spaced bar positions --
        # once a window is narrow enough that real (non-outlier) events pile
        # onto just a handful of adjacent bars, no amount of extra
        # statistics adds more distinct values to fit against.
        return f"only {n_distinct_bins} distinct hodoscope value(s) in this window (< 4 needed)"
    return "fit unreliable (sigma_err too large relative to sigma)"


def window_convergence(x_tracker, x_hodo, widths=DEFAULT_WIDTHS_MM, min_events=DEFAULT_MIN_WINDOW_EVENTS,
                       bins=DEFAULT_BINS, half_width=DEFAULT_HALF_WIDTH_MM):
    """Scan progressively narrower tracker-position windows and fit the
    hodoscope's raw position within each.

    Mismatched (tracker, hodoscope) pairs -- from imperfect tracker-to-ROOT
    alignment, see module docstring -- are clipped out *once*, globally, via
    the full-sample residual (hodo - tracker), the same statistic and clip
    ``fit_resolution`` uses for Method 1. This is deliberate, not
    incidental: an earlier version re-ran ``_robust_clip`` fresh inside each
    width's own small subsample, which turned out to be unsafe on real data
    (run 1774, si2/hodoY) -- a percentile-based threshold computed on a
    heavily bar-quantized sample is extremely sensitive to exactly where a
    percentile's *rank* happens to fall relative to the single dominant
    bar's huge count jump, and a difference of a handful of events between
    two nested windows can flip it to either side. That produced a visibly
    non-monotonic result: a narrower, strictly-nested 0.01mm window
    "succeeded" right where the wider 0.08mm window it's a subset of had
    just failed, purely because each recomputed its own threshold from
    scratch on a different small sample. Clipping once, globally, on the
    full (tens-of-thousands-of-event) sample avoids that instability, and
    every window below is then a guaranteed subset of every wider window --
    genuinely monotonic (up to Poisson noise) rather than only expected to
    be.

    All windows share the same center (the cleaned sample's robust median
    tracker position) -- a single representative spot within the beam
    profile, not a scan across it. See module docstring for why the fitted
    sigma is expected to fall and then plateau as ``widths`` shrinks.

    Returns a list of ``(width, sigma, sigma_err, n_events, fail_reason)`` in
    the same order as ``widths``; ``sigma``/``sigma_err``/``fail_reason`` are
    ``None`` on success. On failure ``fail_reason`` is a short string (see
    ``_diagnose_fit_failure``) -- narrower windows commonly fail simply
    because the hodoscope's own bar quantization runs out of distinct
    values to fit against, not because of a data or fit problem.
    """
    x_tracker = np.asarray(x_tracker)
    x_hodo = np.asarray(x_hodo)
    if len(x_tracker) == 0:
        return [(w, None, None, 0, "no events") for w in widths]

    keep = _robust_clip_mask(x_hodo - x_tracker)
    x_tracker, x_hodo = x_tracker[keep], x_hodo[keep]
    if len(x_tracker) == 0:
        return [(w, None, None, 0, "no events survived the global outlier clip") for w in widths]

    center = np.median(x_tracker)
    results = []
    for w in widths:
        sel = np.abs(x_tracker - center) <= w / 2
        n = int(sel.sum())
        if n < min_events:
            results.append((w, None, None, n, "too few events in this tracker window"))
            continue
        fit = _fit_gaussian(x_hodo[sel], bins=bins, half_width=half_width)
        if fit is None:
            reason = _diagnose_fit_failure(x_hodo[sel], bins=bins, half_width=half_width)
            results.append((w, None, None, n, reason))
            continue
        _, _, sigma, sigma_err, *_ = fit
        results.append((w, sigma, sigma_err, n, None))
    return results


def plot_residual(centers, counts, popt, title, filename, runtype=""):
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    width = centers[1] - centers[0]
    ax.bar(centers, counts, width=width, color="steelblue", alpha=0.8, label="Data")
    xs = np.linspace(centers[0], centers[-1], 400)
    ax.plot(xs, gauss(xs, *popt), color="red", lw=2,
           label=f"$\\mu$ = {popt[1]:.3f} mm\n$\\sigma$ = {abs(popt[2]):.3f} mm")
    ax.set_xlabel("Hodoscope $-$ Tracker [mm]", loc="right")
    ax.set_ylabel("Events", loc="top")
    ax.legend()
    mh.label.exp_label(exp="CaloX", text=runtype, data=True, rlabel=title, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Residual plot saved {filename}")


def plot_convergence(scan, title, filename, runtype=""):
    """Plot fitted sigma vs. tracker-window width, from ``window_convergence``'s output."""
    widths = np.array([w for w, s, se, n, reason in scan])
    sigmas = np.array([s if s is not None else np.nan for w, s, se, n, reason in scan])
    errs = np.array([se if se is not None else 0.0 for w, s, se, n, reason in scan])
    valid = ~np.isnan(sigmas)

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.errorbar(widths[valid], sigmas[valid], yerr=errs[valid], fmt="o-", ms=10,
               color="steelblue", ecolor="black", capsize=4)
    if valid.any():
        w_min = widths[valid].min()
        sigma_at_min = sigmas[valid][np.argmin(widths[valid])]
        ax.axhline(sigma_at_min, color="red", ls="--", lw=1.5,
                  label=f"Smallest usable window ({w_min:g} mm): $\\sigma$ = {sigma_at_min:.3f} mm")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Tracker window width [mm]", loc="right")
    ax.set_ylabel("Fitted hodoscope $\\sigma$ [mm]", loc="top")
    mh.label.exp_label(exp="CaloX", text=runtype, data=True, rlabel=title, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Convergence plot saved {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, default=None,
                        help="Run ID to process (default: every run in run_list.json, pooled)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    parser.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH_MM, metavar="MM",
                        help="Fit window half-width around each distribution's own median, in mm")
    parser.add_argument("--widths", type=float, nargs="+", default=DEFAULT_WIDTHS_MM, metavar="MM",
                        help="Descending tracker-window widths (mm) for the convergence scan")
    parser.add_argument("--min-window-events", type=int, default=DEFAULT_MIN_WINDOW_EVENTS,
                        help="Minimum events required in a tracker window to attempt a fit")
    args = parser.parse_args()

    run_ids = [args.run] if args.run else sorted(load_run_list().keys(), key=int)
    run_label = args.run if args.run else "all_runs"
    runtype = get_beam_label(args.run) if args.run else ""

    pooled = {key: ([], []) for key in LABELS}  # key -> (tracker_parts, hodo_parts)
    for run_id in run_ids:
        try:
            data, match_frac = load_run_data(run_id)
        except Exception as e:
            print(f"[skip] run {run_id}: {e}")
            continue
        for key in LABELS:
            trk, hodo = data[key]
            pooled[key][0].append(trk)
            pooled[key][1].append(hodo)
        if args.run:
            print(f"Run {run_id}: tracker match_frac={match_frac:.4%}  "
                 f"n_hit1={len(data['x1'][0])}  n_hit2={len(data['x2'][0])}")

    pooled = {
        key: (np.concatenate(trk) if trk else np.array([]), np.concatenate(hodo) if hodo else np.array([]))
        for key, (trk, hodo) in pooled.items()
    }

    summary_lines = [f"Hodoscope resolution vs Si Tracker -- {run_label}"]

    # --- Method 1: full-sample residual fit ---
    print(f"\n[Method 1] Full-sample residual fit -- {run_label}")
    header = f"{'Selection':<15}{'mu [mm]':>12}{'sigma [mm]':>14}{'n_used':>10}{'n_total':>10}"
    print(header)
    print("-" * len(header))
    summary_lines += ["", "Method 1: full-sample residual fit", header]

    for key, label in LABELS.items():
        x_tracker, x_hodo = pooled[key]
        residual = x_hodo - x_tracker
        fit = fit_resolution(residual, bins=args.bins, half_width=args.half_width)
        if fit is None:
            line = f"{label:<15}{'--':>12}{'--':>14}{'insufficient data':>20}"
            print(line)
            summary_lines.append(line)
            continue

        mu, mu_err, sigma, sigma_err, n_used, n_total, (centers, counts, popt) = fit
        line = f"{label:<15}{mu:>12.4f}{sigma:>14.4f}{n_used:>10d}{n_total:>10d}"
        print(line)
        summary_lines.append(line)
        summary_lines.append(f"    mu_err = {mu_err:.4f} mm, sigma_err = {sigma_err:.4f} mm")

        filename = os.path.join(OUTPUT_DIR, f"hodores_residual_{key}_{run_label}.png")
        title = f"{label} Residual" if args.run else f"{label} Residual — {run_label}"
        plot_residual(centers, counts, popt, title, filename, runtype=runtype)

    # --- Method 2: tracker-window convergence ---
    print(f"\n[Method 2] Tracker-window convergence -- {run_label}")
    summary_lines += ["", "Method 2: tracker-window convergence (width [mm] -> sigma [mm])"]

    for key, label in LABELS.items():
        x_tracker, x_hodo = pooled[key]
        scan = window_convergence(x_tracker, x_hodo, widths=args.widths,
                                  min_events=args.min_window_events,
                                  bins=args.bins, half_width=args.half_width)

        print(f"\n{label}:")
        summary_lines.append(f"  {label}:")
        row_fmt = "  {:>10}{:>14}{:>14}{:>10}"
        print(row_fmt.format("width[mm]", "sigma[mm]", "sigma_err", "n_events"))
        for w, sigma, sigma_err, n, reason in scan:
            if sigma is None:
                row = row_fmt.format(f"{w:g}", "--", "--", n) + f"   ({reason})"
            else:
                row = row_fmt.format(f"{w:g}", f"{sigma:.4f}", f"{sigma_err:.4f}", n)
            print(row)
            summary_lines.append("    " + row.strip())

        valid = [(w, s, se, n) for w, s, se, n, reason in scan if s is not None]
        if not valid:
            print(f"  {label}: no window had enough statistics to fit")
            summary_lines.append(f"    ({label}: no window had enough statistics to fit)")
            continue

        w_min, sigma_min, sigma_min_err, n_min = min(valid, key=lambda t: t[0])
        msg = f"  -> converged sigma ({label}) = {sigma_min:.4f} +/- {sigma_min_err:.4f} mm at width={w_min:g} mm (n={n_min})"

        skipped_narrower = [(w, reason) for w, s, se, n, reason in scan if s is None and w < w_min]
        if skipped_narrower:
            reasons = {reason for _, reason in skipped_narrower}
            reason_note = reasons.pop() if len(reasons) == 1 else "mixed reasons"
            msg += (f"\n     Note: {len(skipped_narrower)} narrower window(s) could not be fit "
                   f"({reason_note}) -- this may not be the true plateau, just the narrowest fittable point.")

        if len(valid) >= 2:
            w2, s2, se2, n2 = sorted(valid, key=lambda t: t[0])[1]
            if abs(sigma_min - s2) > 2 * np.sqrt(sigma_min_err ** 2 + se2 ** 2):
                msg += ("\n     WARNING: sigma still differs beyond uncertainty between the two smallest "
                       "successfully-fit windows -- not clearly converged.")
        print(msg)
        summary_lines.append(msg)

        filename = os.path.join(OUTPUT_DIR, f"hodores_convergence_{key}_{run_label}.png")
        title = f"{label} Window Convergence" if args.run else f"{label} Window Convergence — {run_label}"
        plot_convergence(scan, title, filename, runtype=runtype)

    summary_path = os.path.join(OUTPUT_DIR, f"hodores_{run_label}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
