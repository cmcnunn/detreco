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
against Si2 (43cm upstream), but 0.26mm against the track (Gaussian-core
widths, before the rotation calibration below). An earlier
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
uncorrected it adds ~0.15mm in quadrature (run 1774, x RMS: 0.253 -> 0.205mm).
Each residual against its *own* coordinate is flat (< 0.5 mrad), so no
scale term is fitted. Validated on held-out events: fitting on even events
flattens the odd events' cross-slopes to < 1 mrad (``plot_rotation_signature``).

This script is where the rotation is *measured*; everything else gets it from
the hodoscope reconstruction. ``--update-run-list`` writes each run's
cross-slopes from a ``--testbeam`` CSV into data/run_list.json
(``hodo_rotation``), and ``utils.hodo.reconstruct_hodoscope(run_id=...)``
applies them. It is not one constant: ~27 mrad for TB2026 runs up to 1820,
~31 mrad from 1829, across the 1824-1828 test runs where the 1cm/3cm
counters went in. This script itself reconstructs the hodoscope without
``run_id`` -- on already-rotated positions ``calibrate_track`` would only
find the leftover, not the rotation.

The width is a 3-sigma-truncated RMS (``truncated_rms``), not a Gaussian
fit: the calibrated residual is flat-topped (a 0.6mm bar blurred by the
track's error), which a Gaussian doesn't describe, and pitch/sqrt(12) at the
lower end of the range below is itself an RMS -- so both ends compare like
with like. The truncation matters: 15-30% of events sit mm away from the
peak (tracker-to-ROOT mismatches, a broad flat background), and a looser
window lets the RMS track alignment quality instead of the hodoscope
(confirmed: with only the 5*MAD clip, run 1812 at 27% mismatched read
0.267mm against 0.224mm for run 1817; truncated at 3 sigma both read
0.197mm). At 160 GeV the 3-sigma window converges to about +/-0.6mm, one
pitch.

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
low energy (sigma_R rises from ~0.20mm at 160 GeV to ~0.24mm at 20 GeV)
and inflates the upper end, so the hodoscope's intrinsic resolution is best
quoted from high-energy runs.

The tracker-to-ROOT alignment is itself imperfect (see utils/tracker.py's
module docstring and scan_alignment_correlation): a run can report a high
match_frac while carrying little genuine per-event correspondence.
``calibrate_track`` refuses a run outright below MIN_ALIGNMENT_CORRELATION
(its per-run fit would otherwise be fitting noise). Its fit, and the RMS's
starting window, begin from a MAD-based clip (same approach as
utils.plotting.draw_fit) so the mismatched events can't drag either off the
real peak.

The hodoscope and tracker each define their own (0, 0), and those origins
are not surveyed to coincide -- confirmed on real data: residuals sit near
a constant few-cm offset (not zero). The calibration absorbs that offset,
and the RMS is taken about the residual's own (truncated) mean rather than
zero -- the width is the target, the mean just a by-product.

Usage:
    python -m scripts.hodores --run <run_id>          # one run
    python -m scripts.hodores --testbeam TB2026       # per-run distribution across a testbeam
    python -m scripts.hodores --update-run-list TB2026   # write that CSV's rotations into run_list.json
    python -m scripts.hodores                         # every run in run_list.json, pooled
"""

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from utils.constants import HG_THRESHOLD, PITCH, X_MAPPING, Y_MAPPING, Z_H, Z_Si1, Z_Si2
from utils.data import RUN_LIST_PATH, get_run_filepath, load_run_list
from utils.hodo import reconstruct_hodoscope
from utils.io import ensure_output_dir
from utils.plotting import get_beam_label, get_run_beam, get_runs_by_testbeam
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
TESTBEAM_OUTPUT_DIR = ensure_output_dir("hodores/testbeam")

# Default --workers for --testbeam: one per CPU this process may use (respects
# a SLURM/cgroup allocation), same as scripts/sihodocor.py. Each worker holds
# one run's ROOT file in memory, so pass fewer on a shared login node.
N_WORKERS = len(os.sched_getaffinity(0))

MAD_CLIP = 5.0  # sigma-equivalent clip: calibration fit sample, and truncated_rms's starting window
RMS_TRUNCATION_SIGMA = 3.0  # truncated_rms keeps events within this many RMS of the mean
MIN_RMS_EVENTS = 20
MIN_CALIBRATION_EVENTS = 200

# Residual plots only: +/-1.5mm in 100 bins (30 um).
PLOT_HALF_WIDTH_MM = 1.5
PLOT_BINS = 100
# --testbeam resolution distribution's bin width.
DISTRIBUTION_BIN_MM = 0.01

# Lower end of the reported range: ideal single-bar (argmax) resolution, see module docstring.
SIGMA_HODO_FLOOR_MM = PITCH / np.sqrt(12)  # 0.173

TRACK_LABELS = {"x": "Track X", "y": "Track Y"}
# x residual / y residual, in every plot.
AXIS_COLORS = {"x": "#2a78d6", "y": "#eb6834"}
# --testbeam's resolution-vs-energy plot: colour + marker per beam type,
# kept off the x/y blue and orange so neither reads as an axis.
BEAM_STYLES = {"pi+": ("#1baf7a", "o"), "mu+": ("#eda100", "s"), "e+": ("#e87ba4", "^")}

# One CSV row per run for --testbeam (see run_resolution).
TESTBEAM_CSV_FIELDS = [
    "run", "beam_type", "beam_energy_gev", "status", "n_both_stations", "match_frac",
    "rotation_mrad", "non_orthogonality_mrad",
    "rms_x_mm", "rms_x_err_mm", "n_used_x", "rms_y_mm", "rms_y_err_mm", "n_used_y", "detail",
]

# --update-run-list: a run's own fit is written only if its residual has a
# real peak (RMS under one pitch -- run 1881, e+ 40 GeV, reads 8.9mm) and its
# non-orthogonality is within this of the testbeam median. The angle between
# the X and Y bar planes is fixed by the hodoscope's construction, so a run
# far off it is a bad fit, not a real change (run 1870: -16 mrad against a
# +5.5 median, while its neighbours sit at +3 to +5).
MAX_NON_ORTHOGONALITY_DEVIATION_MRAD = 10.0
# A run without a trusted fit of its own takes the median of this many nearest
# trusted runs, so one noisy low-statistics fit isn't copied across a long
# stretch (TB2026's 160 tracker-less runs after 1882).
N_NEAREST_TRUSTED = 5

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

    Hodoscope positions are the unrotated bar frame (no ``run_id`` to
    ``reconstruct_hodoscope``): this is the sample the rotation is measured on.

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

    xh, yh, good_hodo = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH)  # unrotated
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

    Starts from ``_robust_clip_mask``'s window, so the iteration begins on
    the real peak rather than the broad mismatch background (see module
    docstring), then repeatedly recomputes the mean and RMS of the kept
    events and keeps everything within ``n_sigma`` RMS of that mean, until
    the kept set stops changing.

    The uncertainty uses the kept sample's own fourth moment,
    var(RMS) ~= (m4 - RMS^4) / (4 RMS^2 N), rather than the Gaussian
    RMS / sqrt(2N): the residual is flat-topped, so the two differ. It is
    statistical only -- the truncation itself is a choice, not a fluctuation.

    Returns ``(mean, rms, rms_err, n_used, n_total, keep)``, or ``None`` if
    fewer than MIN_RMS_EVENTS survive.
    """
    values = np.asarray(values)
    n_total = len(values)
    keep = _robust_clip_mask(values)
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


def plot_track_residual(res_shift, rms_shift, res_cal, rms_cal, axis, title, filename, runtype=""):
    """Residual for one axis: shift-only vs rotation-calibrated, each with its ``truncated_rms``.

    Each is centered on its own truncated mean so the two shapes overlay
    directly; dashed lines mark the calibrated residual's truncation window.
    """
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    edges = np.linspace(-PLOT_HALF_WIDTH_MM, PLOT_HALF_WIDTH_MM, PLOT_BINS + 1)
    for residual, result, color, name in ((res_shift, rms_shift, "gray", "Shift only"),
                                          (res_cal, rms_cal, AXIS_COLORS[axis], "Rotation-calibrated")):
        if result is None:
            continue
        mean, rms, rms_err, *_ = result
        counts, _ = np.histogram(residual - mean, bins=edges)
        ax.stairs(counts, edges, color=color, lw=2.5, label=f"{name}: RMS = {rms:.3f} $\\pm$ {rms_err:.3f} mm")
    if rms_cal is not None:
        window = RMS_TRUNCATION_SIGMA * rms_cal[1]
        ax.axvline(-window, color=AXIS_COLORS[axis], lw=1.5, ls="--",
                   label=f"$\\pm${RMS_TRUNCATION_SIGMA:g} RMS truncation ($\\pm${window:.2f} mm)")
        ax.axvline(window, color=AXIS_COLORS[axis], lw=1.5, ls="--")
    ax.set_xlabel(f"Hodo {axis.upper()} $-$ Track {axis.upper()} at $z_H$ [mm]", loc="right")
    ax.set_ylabel("Events", loc="top")
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    # Opaque, so the truncation lines don't run through the text.
    ax.legend(loc="upper left", fontsize=18, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
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


def run_resolution(run_id):
    """One run's resolution for ``--testbeam``, as one CSV row (see TESTBEAM_CSV_FIELDS).

    The same per-run chain as ``--run`` -- load, calibrate, truncated RMS per
    axis -- without the plots. Never raises: a run that can't be loaded or
    is refused comes back with its reason in "status"/"detail" and the
    resolution columns empty, so one bad run can't take down the pool.
    """
    beam_type, energy = get_run_beam(run_id)
    row = {"run": int(run_id), "beam_type": beam_type or "",
           "beam_energy_gev": energy if energy is not None else ""}
    try:
        # reconstruct_hodoscope's span check warns on every event with an
        # empty plane -- harmless, but it would bury the progress lines.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            track, match_frac = load_run_data(run_id)
    except Exception as e:
        return {**row, "status": "load failed", "detail": str(e)}
    row.update(n_both_stations=len(track["xh"]), match_frac=match_frac)

    cal, reason = calibrate_track(track)
    if cal is None:
        status = "low correlation" if "correlation" in reason else "too few events"
        return {**row, "status": status, "detail": reason}
    a, b = cal["slopes"]
    row.update(rotation_mrad=1e3 * 0.5 * (a - b), non_orthogonality_mrad=1e3 * (a + b))
    for axis in TRACK_LABELS:
        reference, hodo, _ = cal[axis]
        result = truncated_rms(hodo - reference)
        if result is None:
            return {**row, "status": "too few events", "detail": f"truncated RMS failed on {axis}"}
        _, rms, rms_err, n_used, _, _ = result
        row.update({f"rms_{axis}_mm": rms, f"rms_{axis}_err_mm": rms_err, f"n_used_{axis}": n_used})
    return {**row, "status": "ok", "detail": ""}


def _ok_rows(rows):
    return [row for row in rows if row["status"] == "ok"]


def plot_testbeam_distribution(rows, testbeam, filename):
    """Histogram of the per-run resolution (the range's upper end) across a testbeam, X and Y overlaid."""
    ok = _ok_rows(rows)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 9))
    if not ok:
        ax.text(0.5, 0.5, "No runs passed", ha="center", va="center", transform=ax.transAxes)
    else:
        values = {axis: np.array([row[f"rms_{axis}_mm"] for row in ok]) for axis in TRACK_LABELS}
        # Runs above twice the median are left off the plot (the summary and
        # CSV still list them): a run with no real residual peak (confirmed:
        # run 1881, e+ 40 GeV, 8.9mm -- almost all tracker-to-ROOT
        # mismatches, only just past the correlation gate) would otherwise
        # stretch the axis and squash the real runs into one bin.
        cap = 2 * np.median(np.concatenate(list(values.values())))
        shown = {axis: v[v <= cap] for axis, v in values.items()}
        # From just below the pitch/sqrt(12) floor to just past the widest run shown.
        lo = SIGMA_HODO_FLOOR_MM - DISTRIBUTION_BIN_MM
        hi = max(v.max() for v in shown.values()) + 2 * DISTRIBUTION_BIN_MM
        edges = np.arange(lo, hi + DISTRIBUTION_BIN_MM, DISTRIBUTION_BIN_MM)
        for axis, line_style in zip(shown, ("-", "--")):  # dashed Y can't hide X where bins coincide
            ax.hist(shown[axis], bins=edges, histtype="step", lw=2.5, ls=line_style, color=AXIS_COLORS[axis],
                    label=f"{axis.upper()}: median {np.median(values[axis]):.3f} mm")
        ax.axvline(SIGMA_HODO_FLOOR_MM, color="gray", lw=2, ls=":",
                   label=f"pitch/$\\sqrt{{12}}$ = {SIGMA_HODO_FLOOR_MM:.3f} mm (lower end)")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # run counts
        ax.legend(loc="upper right", fontsize=20, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax.set_xlabel(f"Hodoscope resolution, upper end ({RMS_TRUNCATION_SIGMA:g}$\\sigma$-truncated RMS) [mm]",
                  loc="right")
    ax.set_ylabel("Runs", loc="top")
    mh.label.exp_label(ax=ax, exp="CaloX", text=testbeam, rlabel="Hodoscope Resolution Distribution", data=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Resolution distribution saved {filename}")


def plot_testbeam_vs_energy(rows, testbeam, filename):
    """Per-run resolution against beam energy, one panel per axis, coloured by beam type.

    Explains most of the distribution's spread: multiple scattering grows as
    1/E, so the low-energy runs sit well above the high-energy plateau.
    """
    ok = [row for row in _ok_rows(rows) if row["beam_energy_gev"] not in ("", None)]
    plt.style.use(mh.style.ROOT)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
    beam_types = sorted({row["beam_type"] for row in ok}, key=lambda b: (b not in BEAM_STYLES, b))
    # Same 0 to twice-the-median window as plot_testbeam_distribution, so a
    # run with no real residual peak can't flatten every other point.
    hi = 2 * np.median([row[f"rms_{axis}_mm"] for row in ok for axis in TRACK_LABELS]) if ok else 1.0
    for ax, axis in zip(axes, TRACK_LABELS):
        above = [row["run"] for row in ok if row[f"rms_{axis}_mm"] > hi]
        for i, beam in enumerate(beam_types):
            color, marker = BEAM_STYLES.get(beam, ("gray", "D"))
            sel = [row for row in ok if row["beam_type"] == beam and row[f"rms_{axis}_mm"] <= hi]
            # Small per-beam offset so runs at the same energy don't hide each other.
            energy = np.array([float(row["beam_energy_gev"]) for row in sel]) + 1.5 * (i - (len(beam_types) - 1) / 2)
            ax.errorbar(energy, [row[f"rms_{axis}_mm"] for row in sel], yerr=[row[f"rms_{axis}_err_mm"] for row in sel],
                        fmt=marker, ms=10, mew=1.5, color=color, mec="black", ecolor=color, capsize=2,
                        label=f"{beam} ({len(sel)} runs)", zorder=5)
        ax.axhline(SIGMA_HODO_FLOOR_MM, color="gray", lw=2, ls="--",
                   label=f"pitch/$\\sqrt{{12}}$ = {SIGMA_HODO_FLOOR_MM:.3f} mm")
        ax.set_xlabel("Beam energy [GeV]", loc="right")
        ax.set_ylabel(f"Hodoscope {axis.upper()} resolution, upper end [mm]", loc="top")
        ax.set_ylim(0, hi)
        title = f"not shown, above {hi:.2f} mm: run {', '.join(map(str, above))}" if above else None
        ax.legend(loc="lower right", fontsize=16, title=title, title_fontsize=15,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=1)
        mh.label.exp_label(ax=ax, exp="CaloX", text=testbeam, rlabel=f"Hodoscope {axis.upper()}", data=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Resolution vs energy saved {filename}")


def _testbeam_summary(rows, testbeam):
    """Summary text: counts by status, then the resolution's median per (beam type, energy)."""
    ok = _ok_rows(rows)
    floor = SIGMA_HODO_FLOOR_MM
    lines = [f"Hodoscope resolution distribution -- {testbeam} ({len(rows)} runs)",
             f"  Per run: range [pitch/sqrt(12) = {floor:.3f}, {RMS_TRUNCATION_SIGMA:g}-sigma-truncated RMS] mm; "
             "the table gives the upper end."]
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    lines.append("  " + ", ".join(f"{status}: {n}" for status, n in sorted(statuses.items(), key=lambda s: -s[1])))

    header = f"  {'Beam':<10}{'E [GeV]':>8}{'runs':>6}{'median X':>10}{'median Y':>10}{'X 16-84%':>16}{'Y 16-84%':>16}"
    lines += ["", header, "  " + "-" * (len(header) - 2)]

    def row_line(beam, energy, group):
        cells = [f"  {beam:<10}{energy:>8}{len(group):>6}"]
        for axis in TRACK_LABELS:
            cells.append(f"{np.median([row[f'rms_{axis}_mm'] for row in group]):>10.3f}")
        for axis in TRACK_LABELS:
            lo, hi = np.percentile([row[f"rms_{axis}_mm"] for row in group], [16, 84])
            cells.append(f"{f'{lo:.3f}-{hi:.3f}':>16}")
        return "".join(cells)

    groups = {}
    for row in ok:
        groups.setdefault((row["beam_type"] or "unknown", row["beam_energy_gev"]), []).append(row)
    for (beam, energy), group in sorted(groups.items(), key=lambda g: (g[0][0], -float(g[0][1] or 0))):
        lines.append(row_line(beam, f"{energy:g}" if energy != "" else "?", group))
    if ok:
        lines += ["  " + "-" * (len(header) - 2), row_line("all", "", ok)]
    lines.append("  Multiple scattering inflates the upper end at low beam energy -- quote high-energy runs.")
    return lines


def run_testbeam(testbeam, workers=N_WORKERS):
    """Every run in ``testbeam``, in parallel: CSV, resolution distribution, resolution vs energy, summary."""
    runs = get_runs_by_testbeam(testbeam)
    if not runs:
        print(f"No runs found for testbeam '{testbeam}'")
        return
    print(f"Hodoscope resolution for {len(runs)} runs from {testbeam} using {workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_resolution, run): run for run in runs}
        for i, fut in enumerate(as_completed(futures), 1):
            run = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # the worker itself died (e.g. out of memory)
                row = {"run": run, "status": "worker failed", "detail": str(e)}
            rows.append(row)
            if row["status"] == "ok":
                result = f"X {row['rms_x_mm']:.3f}  Y {row['rms_y_mm']:.3f} mm"
            else:
                result = f"{row['status']} ({row.get('detail', '')})"
            print(f"[{i}/{len(runs)}] run {run}: {result}")
    rows.sort(key=lambda row: row["run"])

    tag = testbeam.replace(" ", "_")
    csv_path = os.path.join(TESTBEAM_OUTPUT_DIR, f"{tag}_hodores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TESTBEAM_CSV_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-run CSV saved {csv_path}")

    plot_testbeam_distribution(rows, testbeam, os.path.join(TESTBEAM_OUTPUT_DIR, f"{tag}_resolution.png"))
    plot_testbeam_vs_energy(rows, testbeam, os.path.join(TESTBEAM_OUTPUT_DIR, f"{tag}_resolution_vs_energy.png"))

    summary = _testbeam_summary(rows, testbeam)
    print()
    for line in summary:
        print(line)
    summary_path = os.path.join(TESTBEAM_OUTPUT_DIR, f"{tag}_hodores.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nSummary saved to {summary_path}")


def update_run_list(testbeam):
    """Write each run's rotation from ``run_testbeam``'s CSV into data/run_list.json.

    Every run in the CSV gets a ``hodo_rotation`` key: ``dx_dy_mrad`` and
    ``dy_dx_mrad`` (``calibrate_track``'s a and b) and ``from_runs``. A run
    with a trusted fit (see MAX_NON_ORTHOGONALITY_DEVIATION_MRAD) uses its
    own; any other -- no tracker data, failed alignment, refused by the
    correlation gate -- takes the median of the N_NEAREST_TRUSTED nearest
    trusted runs by run number, which stay on the right side of a
    mid-testbeam change like TB2026's (see module docstring). Runs not in
    the CSV are left untouched.
    """
    tag = testbeam.replace(" ", "_")
    csv_path = os.path.join(TESTBEAM_OUTPUT_DIR, f"{tag}_hodores.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        raise SystemExit(f"No ok runs in {csv_path}; run_list.json not changed")

    median_non_orth = np.median([float(row["non_orthogonality_mrad"]) for row in ok])
    trusted = {}
    for row in ok:
        rotation, non_orth = float(row["rotation_mrad"]), float(row["non_orthogonality_mrad"])
        if max(float(row["rms_x_mm"]), float(row["rms_y_mm"])) >= PITCH:
            print(f"  run {row['run']}: no residual peak (RMS >= {PITCH} mm), not trusted")
            continue
        if abs(non_orth - median_non_orth) > MAX_NON_ORTHOGONALITY_DEVIATION_MRAD:
            print(f"  run {row['run']}: non-orthogonality {non_orth:+.1f} mrad vs median "
                  f"{median_non_orth:+.1f}, not trusted")
            continue
        # rotation = (a - b) / 2, non-orthogonality = a + b (see calibrate_track)
        trusted[int(row["run"])] = (rotation + non_orth / 2, non_orth / 2 - rotation)
    trusted_runs = np.array(sorted(trusted))

    run_list = load_run_list()
    n_own = n_borrowed = 0
    for row in rows:
        run = int(row["run"])
        if str(run) not in run_list:
            continue
        if run in trusted:
            sources = [run]
            n_own += 1
        else:
            nearest = np.argsort(np.abs(trusted_runs - run), kind="stable")[:N_NEAREST_TRUSTED]
            sources = sorted(int(r) for r in trusted_runs[nearest])
            n_borrowed += 1
        dx_dy, dy_dx = np.median([trusted[s] for s in sources], axis=0)
        run_list[str(run)]["hodo_rotation"] = {
            "dx_dy_mrad": round(float(dx_dy), 2), "dy_dx_mrad": round(float(dy_dx), 2), "from_runs": sources,
        }
    with open(RUN_LIST_PATH, "w") as f:
        json.dump(run_list, f, indent=4)
        f.write("\n")
    print(f"Wrote hodo_rotation for {n_own + n_borrowed} {testbeam} runs to {RUN_LIST_PATH}: "
          f"{n_own} from their own fit, {n_borrowed} from the median of the {N_NEAREST_TRUSTED} nearest trusted runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, default=None,
                        help="Run ID to process (default: every run in run_list.json, pooled)")
    parser.add_argument("--testbeam", type=str, default=None,
                        help="Instead of --run, measure every run in this testbeam (e.g. TB2026) separately "
                             "and plot the distribution of per-run resolutions")
    parser.add_argument("--update-run-list", type=str, default=None, metavar="TESTBEAM",
                        help="Write each run's rotation from this testbeam's --testbeam CSV into "
                             "data/run_list.json (hodo_rotation, applied by reconstruct_hodoscope(run_id=...)); "
                             "reprocesses nothing")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"--testbeam: parallel runs (default {N_WORKERS}, every CPU available; "
                             f"each holds one ROOT file in memory)")
    parser.add_argument("--tracker-resolution", type=float, default=None, metavar="MM",
                        help=f"Per-station tracker resolution, if known: adds a point estimate with "
                             f"{TRACK_ERR_GAIN:.2f}x this subtracted in quadrature, alongside the range")
    args = parser.parse_args()

    if args.update_run_list:
        if args.run or args.testbeam:
            parser.error("--update-run-list reads an existing --testbeam CSV; don't combine it with --run/--testbeam")
        update_run_list(args.update_run_list)
        return
    if args.testbeam:
        if args.run:
            parser.error("--run and --testbeam are mutually exclusive")
        run_testbeam(args.testbeam, workers=args.workers)
        return

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

    header = (f"{'Selection':<11}{'RMS shift':>11}{'RMS cal':>10}{'RMS err':>10}"
              f"{'sigma_hodo range':>20}{'n_used':>9}{'n_total':>9}")
    info = f"  RMS = {RMS_TRUNCATION_SIGMA:g}-sigma-truncated RMS of the residual [mm]"
    print(info)
    print(header)
    print("-" * len(header))
    summary_lines += ["", info, header, "-" * len(header)]

    floor = SIGMA_HODO_FLOOR_MM
    upper_ends = {}  # axis -> (rms, rms_err)
    for axis, label in TRACK_LABELS.items():
        reference, hodo, reference_shift = pooled[axis]
        res_shift, res_cal = hodo - reference_shift, hodo - reference
        rms_shift = truncated_rms(res_shift) if len(res_shift) else None
        rms_cal = truncated_rms(res_cal) if len(res_cal) else None
        if rms_cal is None:
            line = f"{label:<11}{'--':>11}{'--':>10}{'insufficient data':>20}"
            print(line)
            summary_lines.append(line)
            continue

        _, rms, rms_err, n_used, n_total, _ = rms_cal
        upper_ends[axis] = (rms, rms_err)
        shift_str = f"{rms_shift[1]:.4f}" if rms_shift is not None else "--"
        range_str = f"{floor:.3f} - {rms:.3f}"
        line = (f"{label:<11}{shift_str:>11}{rms:>10.4f}{rms_err:>10.4f}"
                f"{range_str:>20}{n_used:>9d}{n_total:>9d}")
        print(line)
        summary_lines.append(line)

        filename = os.path.join(OUTPUT_DIR, f"hodores_track_residual_{axis}_{run_label}.png")
        plot_track_residual(res_shift, rms_shift, res_cal, rms_cal, axis, f"{label} Residual", filename,
                            runtype=runtype)

    quote = ["", "Hodoscope resolution (range):"]
    for axis, (rms, rms_err) in upper_ends.items():
        quote.append(f"  {axis.upper()}: {floor:.3f} - {rms:.3f} mm   (upper end +/- {rms_err:.3f} stat.)")
        if rms < floor:
            quote.append("     WARNING: RMS is below pitch/sqrt(12), which argmax reconstruction can't "
                         "reach -- check the calibration")
    quote += [
        f"  upper end = tracker resolution 0 (sigma_hodo = {RMS_TRUNCATION_SIGMA:g}-sigma-truncated RMS)",
        f"  lower end = pitch/sqrt(12) = {floor:.3f} mm, the ideal single-bar resolution",
        "  Multiple scattering inflates the upper end at low beam energy -- quote high-energy runs.",
    ]
    if args.tracker_resolution is not None:
        sigma_track = TRACK_ERR_GAIN * args.tracker_resolution
        quote.append(f"  With --tracker-resolution {args.tracker_resolution:g} mm per station "
                     f"(sigma_track = {sigma_track:.4f} mm):")
        for axis, (rms, _) in upper_ends.items():
            sigma_hodo = np.sqrt(max(rms ** 2 - sigma_track ** 2, 0.0))
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
