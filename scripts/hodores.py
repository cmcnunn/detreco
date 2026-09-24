"""Hodoscope spatial resolution against the silicon tracker.

Compares the hodoscope's reconstructed position to a straight track through
both si-tracker stations, extrapolated to the hodoscope's own z. Reference
selection: hodoscope-good, veto-passing events with a real hit on *both*
stations, matched to ROOT via utils.tracker.build_aligned_tracker_branches
(same as sihodocor.py/sieff.py).

Why a track rather than one station: a single station's residual also
carries the beam's angular spread times the station-to-hodoscope gap (each
particle's angle moves it between the two planes) -- confirmed on real
data, run 1774's x residual is 0.49mm against Si1 (90cm upstream), 0.33mm
against Si2 (43cm upstream), but 0.26mm against the track. An earlier
version of this script ran that single-station fit and a tracker-window
convergence scan too; both were dropped in favour of the track.

With the surveyed z-positions (utils.constants Z_Si1/Z_Si2/Z_H, cm from the
DREAM face), the track is evaluated at the hodoscope's z:

    x_track(Z_H) = x2 * (Z_Si1 - Z_H)/(Z_Si1 - Z_Si2) + x1 * (Z_H - Z_Si2)/(Z_Si1 - Z_Si2)
                 = 1.915 * x2 - 0.915 * x1

The beam passes Si1 -> Si2 -> hodoscope, so this extrapolates past Si2
rather than interpolating. Before fitting, the track is mapped into the
hodoscope frame by a per-run shift + rotation calibration
(``calibrate_track``): the hodoscope is turned ~27 mrad (~1.5 deg) relative
to the tracker, confirmed on every good run checked across pi+/mu+/e+ at
10-160 GeV. The x residual rises with track y (~+30 mrad) and the y
residual falls with track x (~-25 mrad); equal-and-opposite is the
signature of a rotation, and the ~5 mrad left over is a small
non-orthogonality between the X and Y bars, absorbed by the same fit. Left
uncorrected it adds ~0.15mm in quadrature (run 1774, x: 0.257 -> 0.207mm).
Each residual against its *own* coordinate is flat (< 0.5 mrad), so no
scale term is fitted. Validated on held-out events: fitting on even events
flattens the odd events' cross-slopes to < 1 mrad (``plot_rotation_signature``).

What remains is

    sigma_R^2 = sigma_hodo^2 + sigma_track^2 + sigma_MS^2,   sigma_track = 2.12 * sigma_station

(2.12 = sqrt(1.915^2 + 0.915^2), both stations assumed equally precise).
sigma_hodo and sigma_track can't be separated using this data alone: with
three planes on one straight line, each plane's residual against the line
through the other two is the *same* number per event up to a constant
factor (verified on run 1774: event-by-event correlation exactly +/-1), so
a 3x3 "solve for all three resolutions" system is rank 1 -- it only ever
gives one equation. The script therefore reports a *range*:

  - upper end: sigma_track = 0, i.e. sigma_hodo = sigma_R;
  - lower end: pitch/sqrt(12) = 0.173mm, the ideal single-bar resolution.
    ``reconstruct_hodoscope``'s default "argmax" always reports one bar
    centre, and no assignment rule does better than giving each hit the bar
    it actually crossed (a uniform error across one pitch) -- softer bar
    edges only add to it. (Would not hold for its "mean" method, which also
    reports half-pitch positions for two-bar hits.)

``--tracker-resolution`` additionally gives a point estimate from a
per-station value supplied from elsewhere. sigma_MS is not negligible at
low energy (sigma_R rises from ~0.19mm at 160 GeV to ~0.25mm at 20 GeV)
and inflates both ends, so the hodoscope's intrinsic resolution is best
quoted from high-energy runs.

The tracker-to-ROOT alignment is itself imperfect (see utils/tracker.py's
module docstring and scan_alignment_correlation): a run can report a high
match_frac while carrying little genuine per-event correspondence.
``calibrate_track`` refuses a run outright below MIN_ALIGNMENT_CORRELATION
(its per-run fit would otherwise be fitting noise), and every fit is
preceded by a MAD-based clip (same approach as utils.plotting.draw_fit) so
a minority of mismatched events can't drag it off the real peak.

The hodoscope and tracker each define their own (0, 0), and those origins
are not surveyed to coincide -- confirmed on real data: residuals sit near
a constant few-cm offset (not zero). The calibration absorbs that offset,
and the residual fit centers its window on the data's own (robust) median
rather than an assumed absolute range -- sigma is the target, the fitted
mean is just a by-product.

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
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from utils.constants import HG_THRESHOLD, PITCH, X_MAPPING, Y_MAPPING, Z_H, Z_Si1, Z_Si2
from utils.data import get_run_filepath, load_run_list
from utils.fit_funcs import gauss
from utils.hodo import reconstruct_hodoscope
from utils.io import ensure_output_dir
from utils.plotting import get_beam_label
from utils.selectors import get_branch_names, passes_veto
from utils.tracker import (
    MIN_ALIGNMENT_CORRELATION,
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)

OUTPUT_DIR = ensure_output_dir("hodores")

# The calibrated residual is ~0.2mm wide and flat-topped (bar quantization),
# which makes the Gaussian sigma sensitive to coarse bins -- 0.1mm bins read
# ~1.5% high on run 1774, while anything from 15 to 50 um bins agrees to
# < 1%. +/-1.5mm with 100 bins gives 30 um.
DEFAULT_HALF_WIDTH_MM = 1.5  # residual-fit window is [median - this, median + this]
DEFAULT_BINS = 100
MAD_CLIP = 5.0  # sigma-equivalent clip applied before every fit below
MAX_RELATIVE_SIGMA_ERR = 0.5  # reject a fit if sigma_err exceeds this fraction of sigma
MIN_CALIBRATION_EVENTS = 200

# Lower end of the reported range: ideal single-bar (argmax) resolution, see module docstring.
SIGMA_HODO_FLOOR_MM = PITCH / np.sqrt(12)  # 0.173

TRACK_LABELS = {"x": "Track X", "y": "Track Y"}
# x residual / y residual, in every plot.
AXIS_COLORS = {"x": "#2a78d6", "y": "#eb6834"}

# Straight-line weights putting the two-station track at the hodoscope's z
# (see module docstring): x_track = TRACK_W1 * x1 + TRACK_W2 * x2.
TRACK_W1 = (Z_H - Z_Si2) / (Z_Si1 - Z_Si2)  # -0.915
TRACK_W2 = (Z_Si1 - Z_H) / (Z_Si1 - Z_Si2)  # +1.915
# Per-station tracker error -> error on x_track, propagated through the weights.
TRACK_ERR_GAIN = float(np.hypot(TRACK_W1, TRACK_W2))  # 2.12


def load_run_data(run_id):
    """Return ``(track, match_frac)`` for one run.

    ``track`` is a dict of same-length mm arrays "x1", "y1", "x2", "y2"
    (tracker) and "xh", "yh" (hodoscope), for the reference-selected events
    (good hodoscope hit, veto pass) that also registered a real hit on
    *both* stations -- no sentinel rows.

    Tracker positions are the raw hardware scale (x10 cm -> mm), not
    ``utils.tracker.calibrate_tracker_positions``'s per-station Hodo-vs-Tracker
    slopes: those were fitted one station at a time, so they partly absorb the
    beam's angular spread over each station's own gap -- exactly the term the
    extrapolation already removes. The track's own-coordinate residual slope
    is flat to < 0.5 mrad on the raw scale.

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

    both = (good_hodo & veto_sel & (x1 != SENTINEL_X1) & (y1 != SENTINEL_Y1)
            & (x2 != SENTINEL_X2) & (y2 != SENTINEL_Y2))
    # tracker x1/y1/x2/y2 are in cm; *10 puts them in mm alongside xh/yh.
    track = {
        "x1": 10 * x1[both], "y1": 10 * y1[both], "x2": 10 * x2[both], "y2": 10 * y2[both],
        "xh": xh[both], "yh": yh[both],
    }
    return track, match_frac


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

    Low-level piece of ``fit_resolution``, which clips first. The histogram
    window is centered on the data's own median rather than a fixed
    absolute range (see module docstring).

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
    own statistics -- fine for the large, stable full-run residuals fit here.

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


def extrapolate_to_hodo(t1, t2):
    """Straight line through the two station hits, evaluated at the hodoscope's z (``Z_H``)."""
    return TRACK_W1 * np.asarray(t1) + TRACK_W2 * np.asarray(t2)


def calibrate_track(track, fit_mask=None):
    """Per-run shift + rotation calibration of the extrapolated track into the hodoscope frame.

    ``track`` is ``load_run_data``'s both-station sample. After the usual
    robust clip, fits one line per axis to the residual against the *other*
    coordinate:

        hodo_x - x_track = a * y_track + c_x
        hodo_y - y_track = b * x_track + c_y

    A rigid rotation by theta shows up as a = +theta, b = -theta (the X bars
    lean one way as you go up in y, the Y bars the other way as you go
    across in x), so the rotation is (a - b) / 2; a + b is the
    non-orthogonality between the X and Y bars, which the same two lines
    absorb. The calibrated reference is then ``x_track + a * y_track + c_x``
    (likewise y) -- the track as the hodoscope's frame sees it.

    ``fit_mask`` (bool, one per event) restricts which events the fit uses
    while still applying the result to every event -- for testing the
    calibration on events it never saw (see ``plot_rotation_signature``), since
    on its own fit sample the calibrated cross-slope is zero by construction.

    Returns ``(cal, None)`` on success, or ``(None, reason)`` if the run is
    refused: too few both-station events, or a hodoscope-vs-track Pearson r
    below MIN_ALIGNMENT_CORRELATION -- a tracker-to-ROOT alignment that
    isn't finding real correspondences (confirmed: run 1866, e+ 40 GeV, r ~
    0), where this fit would only be fitting noise. ``cal`` holds, per axis
    ("x"/"y"), ``(reference, hodo, reference_shift_only)`` same-length mm
    arrays, plus "slopes" ``(a, b)`` in rad, "r", and -- for
    ``plot_rotation_signature`` -- "track_x"/"track_y" (uncalibrated) and
    "keep" (the clip mask the fit used).
    """
    px = extrapolate_to_hodo(track["x1"], track["x2"])
    py = extrapolate_to_hodo(track["y1"], track["y2"])
    xh, yh = track["xh"], track["yh"]
    if len(px) < MIN_CALIBRATION_EVENTS:
        return None, f"only {len(px)} both-station events (< {MIN_CALIBRATION_EVENTS})"
    r = min(np.corrcoef(xh, px)[0, 1], np.corrcoef(yh, py)[0, 1])
    if not r >= MIN_ALIGNMENT_CORRELATION:  # nan-safe
        return None, f"hodoscope-vs-track correlation r={r:.2f} < {MIN_ALIGNMENT_CORRELATION}"

    keep = _robust_clip_mask(xh - px) & _robust_clip_mask(yh - py)
    fit = keep if fit_mask is None else keep & fit_mask
    a, cx = np.polyfit(py[fit], (xh - px)[fit], 1)
    b, cy = np.polyfit(px[fit], (yh - py)[fit], 1)
    cal = {
        "x": (px + a * py + cx, xh, px + np.median((xh - px)[fit])),
        "y": (py + b * px + cy, yh, py + np.median((yh - py)[fit])),
        "slopes": (a, b), "r": r, "track_x": px, "track_y": py, "keep": keep,
    }
    return cal, None


def plot_track_residual(fit_shift, fit_cal, axis, title, filename, runtype=""):
    """Residual for one axis: shift-only vs rotation-calibrated, each with its Gaussian fit."""
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    for fit, color, name in ((fit_shift, "gray", "Shift only"), (fit_cal, AXIS_COLORS[axis], "Rotation-calibrated")):
        if fit is None:
            continue
        mu, _, sigma, sigma_err, _, _, (centers, counts, popt) = fit
        # Center each on its own fitted mean so the two shapes overlay directly.
        half_bin = 0.5 * (centers[1] - centers[0])
        edges = np.append(centers - half_bin, centers[-1] + half_bin) - mu
        ax.stairs(counts, edges, color=color, lw=2.5,
                  label=f"{name}: $\\sigma$ = {sigma:.3f} $\\pm$ {sigma_err:.3f} mm")
        xs = np.linspace(centers[0], centers[-1], 400)
        ax.plot(xs - mu, gauss(xs, *popt), color=color, lw=1.5, ls="--")
    ax.set_xlabel(f"Hodo {axis.upper()} $-$ Track {axis.upper()} at $z_H$ [mm]", loc="right")
    ax.set_ylabel("Events", loc="top")
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.legend(loc="upper left", fontsize=18)
    mh.label.exp_label(exp="CaloX", text=runtype, data=True, rlabel=title, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Track residual plot saved {filename}")


def _profile(x, y, n_bins=16, min_count=30):
    """Mean of ``y`` (and its standard error) in equal-population bins of ``x``."""
    edges = np.percentile(x, np.linspace(1, 99, n_bins + 1))
    centers, means, errs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < min_count:
            continue
        centers.append(x[sel].mean())
        means.append(y[sel].mean())
        errs.append(y[sel].std() / np.sqrt(sel.sum()))
    return np.array(centers), np.array(means), np.array(errs)


def _draw_profile_fit(ax, x, y, x_range, name, marker, point_fill, line_color, line_style):
    """One profile (black-edged points) + its straight-line fit on ``ax``, slope in the legend."""
    centers, means, errs = _profile(x, y)
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    ax.errorbar(centers, means, yerr=errs, fmt=marker, ms=9, color="black", mfc=point_fill, mew=2, zorder=5,
               label=f"{name}slope = {1e3 * slope:+.1f} $\\pm$ {1e3 * np.sqrt(cov[0, 0]):.1f} mrad")
    xs = np.linspace(*x_range, 2)
    ax.plot(xs, slope * xs + intercept, color=line_color, ls=line_style, lw=2.5, zorder=4)


def plot_rotation_signature(cal, filename, runtype="", rlabel="", test_mask=None):
    """Evidence for ``calibrate_track``'s rotation.

    Top row: each residual against the *other* coordinate -- the rotation
    signature (equal-and-opposite slopes, see ``calibrate_track``). Bottom
    row: against its own coordinate -- the scale control, which should be
    flat; the sawtooth there is the hodoscope reporting bar centres as the
    track crosses each 0.6mm bar.

    With ``test_mask`` omitted, plots the *uncalibrated* (shift-only)
    residuals: the evidence that a rotation is there. With ``test_mask``
    given, ``cal`` should come from ``calibrate_track(track,
    fit_mask=~test_mask)``, and only the ``test_mask`` events -- ones the
    calibration never saw -- are plotted: the 2D histogram is the calibrated
    residual, with the same events' shift-only profile overlaid in gray. On
    its own fit sample the calibrated cross-slope is zero by construction,
    so only a held-out sample actually tests it. The top row should go flat,
    and the bottom row's sawtooth teeth sharpen, since the rotation no longer
    smears each bar edge across the other coordinate.
    """
    sel = cal["keep"] if test_mask is None else cal["keep"] & test_mask
    before = {axis: (cal[axis][1] - cal[axis][2])[sel] for axis in TRACK_LABELS}
    after = {axis: (cal[axis][1] - cal[axis][0])[sel] for axis in TRACK_LABELS}
    if test_mask is None:
        px, py = cal["track_x"][sel], cal["track_y"][sel]
        shown, frame, suffix = before, "at hodoscope", ""
    else:
        # The calibrated track, i.e. the hodoscope's own frame: there the bar
        # edges sit at fixed positions, so the bottom row's teeth can sharpen.
        # Against the raw track position they'd instead blur, since each edge
        # lands at a different raw x for every y once the rotation is removed.
        px, py = cal["x"][0][sel], cal["y"][0][sel]
        shown, frame, suffix = after, "(hodoscope frame)", ", after calibration (held-out events)"

    plt.style.use(mh.style.ROOT)
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    panels = [
        (axes[0, 0], py, "x", "Track Y", "Hodo X $-$ Track X", "rotation signature"),
        (axes[0, 1], px, "y", "Track X", "Hodo Y $-$ Track Y", "rotation signature"),
        (axes[1, 0], px, "x", "Track X", "Hodo X $-$ Track X", "control (own coordinate)"),
        (axes[1, 1], py, "y", "Track Y", "Hodo Y $-$ Track Y", "control (own coordinate)"),
    ]
    for ax, x, axis, xlabel, ylabel, what in panels:
        x_range = np.percentile(x, [0.5, 99.5])
        ax.hist2d(x, shown[axis], bins=[80, 60], range=[x_range, [-1.5, 1.5]], cmap="Blues", norm=LogNorm(),
                  rasterized=True)
        if test_mask is None:
            _draw_profile_fit(ax, x, before[axis], x_range, "", "o", "white", AXIS_COLORS[axis], "-")
        else:
            _draw_profile_fit(ax, x, before[axis], x_range, "Before: ", "s", "lightgray", "gray", "--")
            _draw_profile_fit(ax, x, after[axis], x_range, "After: ", "o", "white", AXIS_COLORS[axis], "-")
        ax.axhline(0, color="gray", lw=1, ls=":")
        ax.set_xlabel(f"{xlabel} {frame} [mm]", loc="right")
        ax.set_ylabel(f"{ylabel} [mm]", loc="top")
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc="upper left", fontsize=20, title=what + suffix, title_fontsize=20)
        mh.label.exp_label(exp="CaloX", text=runtype, data=True, rlabel=rlabel, ax=ax, fontsize=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Rotation signature plot saved {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, default=None,
                        help="Run ID to process (default: every run in run_list.json, pooled)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    parser.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH_MM, metavar="MM",
                        help="Fit window half-width around each residual's own median, in mm")
    parser.add_argument("--tracker-resolution", type=float, default=None, metavar="MM",
                        help=f"Per-station tracker resolution, if known: adds a point estimate with "
                             f"{TRACK_ERR_GAIN:.2f}x this subtracted in quadrature, alongside the range")
    args = parser.parse_args()

    run_ids = [args.run] if args.run else sorted(load_run_list().keys(), key=int)
    run_label = args.run if args.run else "all_runs"
    runtype = get_beam_label(args.run) if args.run else ""
    rlabel = f"Run {run_label}" if args.run else run_label

    # axis -> (reference_parts, hodo_parts, reference_shift_only_parts), each
    # already in its own run's hodoscope frame, so pooling across runs is safe.
    pooled = {axis: ([], [], []) for axis in TRACK_LABELS}
    calibrations = []  # (run_id, a, b) per accepted run
    n_refused = 0
    last_cal = last_track = None
    for run_id in run_ids:
        try:
            track, match_frac = load_run_data(run_id)
        except Exception as e:
            print(f"[skip] run {run_id}: {e}")
            continue
        if args.run:
            print(f"Run {run_id}: tracker match_frac={match_frac:.4%}  n_both_stations={len(track['xh'])}")

        cal, reason = calibrate_track(track)
        if cal is None:
            n_refused += 1
            print(f"[skip] run {run_id}: {reason}")
            continue
        for axis in TRACK_LABELS:
            for parts, arr in zip(pooled[axis], cal[axis]):
                parts.append(arr)
        calibrations.append((run_id, *cal["slopes"]))
        last_cal, last_track = cal, track

    pooled = {
        axis: tuple(np.concatenate(parts) if parts else np.array([]) for parts in arrays)
        for axis, arrays in pooled.items()
    }

    header_line = f"Hodoscope resolution vs Si Tracker -- {run_label}"
    print(f"\n{header_line}")
    summary_lines = [header_line]
    info = [f"  Track extrapolated to hodoscope z: x_track(z_H) = {TRACK_W2:.3f}*x2 {TRACK_W1:+.3f}*x1   "
            f"(z from DREAM face: Si1={Z_Si1:g}, Si2={Z_Si2:g}, hodo={Z_H:g} cm)"]
    if calibrations:
        a = np.array([c[1] for c in calibrations])
        b = np.array([c[2] for c in calibrations])
        rotation, non_orth = 0.5 * (a - b), a + b
        if args.run:
            info.append(f"  Rotation calibration: dX/dY = {1e3 * a[0]:+.1f} mrad, dY/dX = {1e3 * b[0]:+.1f} mrad"
                        f"  ->  rotation {1e3 * rotation[0]:.1f} mrad, non-orthogonality {1e3 * non_orth[0]:+.1f} mrad")
        else:
            info.append(f"  Rotation calibration over {len(calibrations)} run(s): rotation median "
                        f"{1e3 * np.median(rotation):.1f} mrad (range {1e3 * rotation.min():.1f} to "
                        f"{1e3 * rotation.max():.1f}), non-orthogonality median {1e3 * np.median(non_orth):+.1f} mrad")
    if n_refused:
        info.append(f"  {n_refused} run(s) refused -- see the [skip] lines above")
    for line in info:
        print(line)
    summary_lines += info

    header = (f"{'Selection':<11}{'sigma_R shift':>15}{'sigma_R cal':>13}{'sigma_err':>11}"
              f"{'sigma_hodo range':>20}{'n_used':>9}{'n_total':>9}")
    print(header)
    print("-" * len(header))
    summary_lines += ["", header, "-" * len(header)]

    floor = SIGMA_HODO_FLOOR_MM
    upper_ends = {}  # axis -> (sigma_R, sigma_err)
    for axis, label in TRACK_LABELS.items():
        reference, hodo, reference_shift = pooled[axis]
        fit_shift = fit_resolution(hodo - reference_shift, bins=args.bins, half_width=args.half_width)
        fit_cal = fit_resolution(hodo - reference, bins=args.bins, half_width=args.half_width)
        if fit_cal is None:
            line = f"{label:<11}{'--':>15}{'--':>13}{'insufficient data':>20}"
            print(line)
            summary_lines.append(line)
            continue

        _, _, sigma, sigma_err, n_used, n_total, _ = fit_cal
        upper_ends[axis] = (sigma, sigma_err)
        shift_str = f"{fit_shift[2]:.4f}" if fit_shift is not None else "--"
        range_str = f"{floor:.3f} - {sigma:.3f}"
        line = (f"{label:<11}{shift_str:>15}{sigma:>13.4f}{sigma_err:>11.4f}"
                f"{range_str:>20}{n_used:>9d}{n_total:>9d}")
        print(line)
        summary_lines.append(line)

        filename = os.path.join(OUTPUT_DIR, f"hodores_track_residual_{axis}_{run_label}.png")
        plot_track_residual(fit_shift, fit_cal, axis, f"{label} Residual", filename, runtype=runtype)

    quote = ["", "Hodoscope resolution (range):"]
    for axis, (sigma, sigma_err) in upper_ends.items():
        quote.append(f"  {axis.upper()}: {floor:.3f} - {sigma:.3f} mm   (upper end +/- {sigma_err:.3f} stat.)")
        if sigma < floor:
            quote.append(f"     WARNING: sigma_R is below pitch/sqrt(12) -- the Gaussian core is "
                         f"underestimating this flat-topped residual")
    quote += [
        f"  upper end = tracker resolution 0 (sigma_hodo = sigma_R)",
        f"  lower end = pitch/sqrt(12) = {floor:.3f} mm, the ideal single-bar resolution",
        "  Multiple scattering inflates the upper end at low beam energy -- quote high-energy runs.",
    ]
    if args.tracker_resolution is not None:
        sigma_track = TRACK_ERR_GAIN * args.tracker_resolution
        quote.append(f"  With --tracker-resolution {args.tracker_resolution:g} mm per station "
                     f"(sigma_track = {sigma_track:.4f} mm):")
        for axis, (sigma, _) in upper_ends.items():
            sigma_hodo = np.sqrt(max(sigma ** 2 - sigma_track ** 2, 0.0))
            note = "  -- below pitch/sqrt(12), so this tracker resolution is too large" if sigma_hodo < floor else ""
            quote.append(f"    {axis.upper()}: {sigma_hodo:.3f} mm{note}")
    for line in quote:
        print(line)
    summary_lines += quote

    if args.run and last_cal is not None:
        filename = os.path.join(OUTPUT_DIR, f"hodores_rotation_{run_label}.png")
        plot_rotation_signature(last_cal, filename, runtype=runtype, rlabel=rlabel)

        # Same panels after calibration, fit on even events and shown on odd
        # ones -- interleaved in time, so both halves see the same conditions.
        odd = np.arange(len(last_track["xh"])) % 2 == 1
        holdout_cal, _ = calibrate_track(last_track, fit_mask=~odd)
        filename = os.path.join(OUTPUT_DIR, f"hodores_rotation_calibrated_{run_label}.png")
        plot_rotation_signature(holdout_cal, filename, runtype=runtype, rlabel=rlabel, test_mask=odd)

    summary_path = os.path.join(OUTPUT_DIR, f"hodores_{run_label}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
