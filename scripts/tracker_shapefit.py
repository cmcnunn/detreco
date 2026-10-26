"""Reconstruct the veto's and 1cm/3cm test-counters' known physical
dimensions from one run's data, using each of the silicon tracker's two
stations and the hodoscope in turn as the reference position -- three
independent "rulers" against which to check a fitted dimension.

For each reference (tracker station 1, tracker station 2, hodoscope), the
same recipe as ``scripts/effplots.py``: the reference detector's own (x, y)
as the denominator, veto-passing or counter-hit events as the numerator.
Instead of just plotting the resulting 2D efficiency map, this fits it to a
closed-form geometric model (utils.fit_funcs.erf_disk / erf_box) and
reports the fitted radius/side against the known nominal value.

  Veto (round, nominal radius VETO_RADIUS_MM = 25mm): a hard circular
  boundary convolved with Gaussian smearing (reference detector's own
  resolution + multiple scattering) is the standard radial edge-response
  turn-on:
      eff(x, y) = eff0 * 0.5 * (1 - erf((r - R) / (sqrt(2) * sigma)))

  1cm/3cm test counters (square, nominal half-side COUNTER_1CM_HALF_SIDE_MM
  = 5mm / COUNTER_3CM_HALF_SIDE_MM = 15mm, assumed axis-aligned): product of
  two independent smeared step-edges, one per axis. hx/hy are fit
  independently rather than forced equal, so a real x/y scale mismatch (or
  the counter's known non-square footprint -- see
  [[project_1cm_counter_geometry]] / utils.fit_funcs erf_box's docstring)
  shows up as hx != hy instead of being averaged away.

If a reference detector's own position reconstruction is correct, the
fitted numbers should recover the known physical dimensions to within the
fit's smearing-limited precision. A discrepancy that shows up against one
reference but not the other two points at that one detector; a
discrepancy that's consistent across all three points at the counter's own
geometry differing from the assumed nominal value instead.

Tracker1/Tracker2 positions come back already calibrated into the
hodoscope's frame -- ``build_aligned_tracker_branches(..., run_id=...)``
applies utils.tracker.get_tracker_hodo_calibration's fitted Hodo-vs-Tracker
slope and offset (this run's own, from
``data/tracker_hodo_slopes/<testbeam>_slopes.csv`` -- see
``scripts.sihodocor --testbeam``; falling back to that testbeam's median, or
no correction) as a reconstruction step, so a tracker-referenced fit
reports a size, and a centre, in the same coordinates as the
hodoscope-referenced fit -- without this, a run's tracker-vs-hodo scale
mismatch alone (not any real physical difference) would show up as the
tracker references disagreeing with the hodoscope reference on the
veto/counter dimensions.

Usage:
    python -m scripts.tracker_shapefit --run 1832
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import HG_THRESHOLD, PITCH, VETO_RADIUS_MM, VETO_THRESHOLD, X_MAPPING, Y_MAPPING
from utils.data import get_run_filepath
from utils.fit_funcs import erf_box, erf_disk
from utils.hodo import hodo_bin_edges, reconstruct_hodoscope
from utils.io import ensure_output_dir
from utils.plotting import MAP_CMAP, NO_DATA_COLOR, _hist_edges, get_beam_label, run_header_label
from utils.selectors import (
    COUNTER_1CM_3CM_FIRST_RUN,
    counter_1cm_hit_mask,
    counter_3cm_hit_mask,
    get_branch_names,
    get_counter_branch_names,
    passes_veto,
)
from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)

# --- Hodoscope HG branch names (matches scripts/effplots.py) ---
Y_HG_BRANCH = "FERS_Board0_energyHG"
X_HG_BRANCH = "FERS_Board1_energyHG"

# --- Known physical dimensions (mm) ---
# VETO_RADIUS_MM lives in utils/constants.py (shared with scripts/energy_tracks.py).
COUNTER_1CM_HALF_SIDE_MM = 5.0
COUNTER_3CM_HALF_SIDE_MM = 15.0

# Margin (mm) of the *_fit_*.png plot crop around the fitted shape, on top
# of its fitted radius/half-sides -- same idea as energy_tracks.py's
# veto-shape zoom, applied here directly from this fit's own center and
# size instead of a hardcoded box. This crops the *plot* only; the fit_*
# functions' own windows (fitted size + a larger margin) are left alone,
# since that extra margin is real data the fit needs to see the eff=0
# plateau and constrain the sigmoid edge.
FIT_PLOT_MARGIN_MM = 5.0

# Bin width (mm) the fit panel's model is drawn on -- it's a closed-form
# function, so there's no reason to show it pixelated at the data's binning.
MODEL_PLOT_BIN_MM = 0.1

# Tracker -> hodoscope frame calibration (fitted Hodo-vs-Tracker slope and
# offset, per-run with a testbeam-median fallback) lives in utils/tracker.py --
# build_aligned_tracker_branches applies it directly when given a run_id
# (see its and utils.tracker.get_tracker_hodo_calibration's docstrings),
# since every consumer of tracker positions benefits from it, not just this
# script.
# --- Bin widths for the efficiency-map grids ---
VETO_BIN_MM = 1.0
COUNTER_BIN_MM = 0.5

# Minimum reference hits for a bin to enter the fit (see fit_shape).
MIN_REF_COUNT = 10

# A fitted parameter's uncertainty this large (mm) means curve_fit's
# covariance blew up -- the data don't constrain it at all (e.g. the
# reference detector's own illuminated region doesn't reach far enough for
# that edge to ever appear in the map).
UNCONSTRAINED_ERR_MM = 100.0

# Added in quadrature to each bin's binomial uncertainty. The naive
# sqrt(eff*(1-eff)/n) formula goes to zero for an exactly-0-or-1 bin
# regardless of n, letting a handful of such bins dominate the fit with
# unphysically tight weight.
SYSTEMATIC_ERR_FLOOR = 0.02


def _aligned_edges(x_ref, x_range, bin_mm):
    """Bin edges for ``x_ref`` restricted to ``x_range``, snapped to the
    reference detector's own real position values when it only takes a
    handful of exact ones (e.g. the hodoscope's bar-pitch-quantized
    positions) rather than an arbitrary evenly-spaced grid.

    An arbitrary linspace grid generally doesn't line up with that
    quantization, so its bin phase drifts across the window and some bins
    land entirely between two real values -- a "dead" bin reading zero
    reference hits even deep inside a well-illuminated region, not a real
    efficiency drop (the same failure mode ``scripts/effplots.py``
    documents and avoids for its own hodoscope-referenced maps). Positions
    that don't repeat exact values (e.g. the silicon tracker's, effectively
    continuous) fall back to a plain evenly-spaced grid via
    ``utils.plotting._hist_edges``. The hodoscope reference itself no longer
    comes through here: rotated into the tracker frame its positions aren't
    exact bar values, so ``fit_shape(hodo=True)`` uses
    ``utils.hodo.hodo_bin_edges`` instead.
    """
    n_bins = max(4, int(round((x_range[1] - x_range[0]) / bin_mm)))
    in_window = x_ref[(x_ref >= x_range[0]) & (x_ref <= x_range[1])]
    if len(in_window) < 2:
        return np.linspace(x_range[0], x_range[1], n_bins + 1)
    # _hist_edges snaps to real values when there are few enough unique
    # ones to fit within n_bins (quantized data, e.g. the hodoscope); with
    # many more unique values than n_bins (continuous data, e.g. the
    # silicon tracker) it falls back to its own evenly-spaced grid.
    return _hist_edges(in_window, bins=n_bins)


def fit_shape(x_ref, y_ref, x_sel, y_sel, model, p0, bounds, x_range, y_range, bin_mm,
              min_ref_count=MIN_REF_COUNT, hodo=False):
    """Fit ``model`` to the (x_ref, y_ref, x_sel, y_sel) efficiency map.

    Only bins with at least ``min_ref_count`` reference hits enter the fit,
    but the returned ``eff`` map covers every bin with at least one (NaN
    only where there are none), so the plots show the whole footprint.
    With ``hodo=True`` (hodoscope reference) the grid is one bar pitch per
    bin and ``bin_mm`` is ignored.

    Returns ``(popt, perr, (eff, h_ref, xedges, yedges))``, or ``None`` if
    there aren't enough well-populated bins to fit.
    """
    if hodo:
        xedges, yedges = hodo_bin_edges(*x_range), hodo_bin_edges(*y_range)
    else:
        xedges = _aligned_edges(x_ref, x_range, bin_mm)
        yedges = _aligned_edges(y_ref, y_range, bin_mm)
    h_ref, xedges, yedges = np.histogram2d(x_ref, y_ref, bins=[xedges, yedges])
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[xedges, yedges])
    eff = np.divide(h_sel, h_ref, out=np.full_like(h_sel, np.nan, dtype=float), where=h_ref > 0)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # A low-count bin's binomial error is unreliable (a 1-hit bin reads
    # exactly 0 or 1 with only the SYSTEMATIC_ERR_FLOOR as its error), so
    # those bins are left out of the fit.
    finite = h_ref >= min_ref_count
    if finite.sum() < len(p0) + 1:
        return None

    n = h_ref[finite]
    e = np.clip(eff[finite], 0.0, 1.0)
    err = np.sqrt(e * (1 - e) / n + SYSTEMATIC_ERR_FLOOR ** 2)

    popt, pcov = curve_fit(
        model, (X[finite], Y[finite]), e, p0=p0, sigma=err,
        absolute_sigma=True, bounds=bounds, maxfev=40000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, (eff, h_ref, xedges, yedges)


def fit_veto(x_ref, y_ref, x_sel, y_sel, hodo=False):
    if len(x_sel) == 0:
        return None
    x0, y0 = float(np.median(x_sel)), float(np.median(y_sel))
    margin = 20.0
    half_window = VETO_RADIUS_MM + margin
    x_range = (x0 - half_window, x0 + half_window)
    y_range = (y0 - half_window, y0 + half_window)
    p0 = [x0, y0, VETO_RADIUS_MM, 2.0, 0.9]
    bounds = ([x0 - margin, y0 - margin, 5.0, 0.1, 0.0],
              [x0 + margin, y0 + margin, 60.0, 20.0, 1.2])
    return fit_shape(x_ref, y_ref, x_sel, y_sel, erf_disk, p0, bounds,
                     x_range, y_range, VETO_BIN_MM, hodo=hodo)


def fit_counter_1cm(x_ref, y_ref, x_sel, y_sel, hodo=False):
    if len(x_sel) == 0:
        return None
    x0, y0 = float(np.median(x_sel)), float(np.median(y_sel))
    margin = 10.0
    half_window = COUNTER_1CM_HALF_SIDE_MM + margin
    x_range = (x0 - half_window, x0 + half_window)
    y_range = (y0 - half_window, y0 + half_window)
    p0 = [x0, y0, COUNTER_1CM_HALF_SIDE_MM, COUNTER_1CM_HALF_SIDE_MM, 1.0, 1.0, 0.7]
    max_half_side = 2 * COUNTER_1CM_HALF_SIDE_MM + margin
    bounds = ([x0 - margin, y0 - margin, 0.5, 0.5, 0.05, 0.05, 0.0],
              [x0 + margin, y0 + margin, max_half_side, max_half_side, 10.0, 10.0, 1.2])
    return fit_shape(x_ref, y_ref, x_sel, y_sel, erf_box, p0, bounds,
                     x_range, y_range, COUNTER_BIN_MM, hodo=hodo)


def fit_counter_3cm(x_ref, y_ref, x_sel, y_sel, hodo=False):
    if len(x_sel) == 0:
        return None
    x0, y0 = float(np.median(x_sel)), float(np.median(y_sel))
    margin = 10.0
    half_window = COUNTER_3CM_HALF_SIDE_MM + margin
    x_range = (x0 - half_window, x0 + half_window)
    y_range = (y0 - half_window, y0 + half_window)
    p0 = [x0, y0, COUNTER_3CM_HALF_SIDE_MM, COUNTER_3CM_HALF_SIDE_MM, 1.0, 1.0, 0.7]
    max_half_side = 2 * COUNTER_3CM_HALF_SIDE_MM + margin
    bounds = ([x0 - margin, y0 - margin, 0.5, 0.5, 0.05, 0.05, 0.0],
              [x0 + margin, y0 + margin, max_half_side, max_half_side, 10.0, 10.0, 1.2])
    return fit_shape(x_ref, y_ref, x_sel, y_sel, erf_box, p0, bounds,
                     x_range, y_range, COUNTER_BIN_MM, hodo=hodo)


def _is_constrained(err):
    # err <= 0 is just as degenerate as err too large: it means curve_fit's
    # Jacobian was singular in this direction (zero sensitivity), not a
    # genuinely perfect measurement.
    return np.isfinite(err) and 0 < err <= UNCONSTRAINED_ERR_MM


def _param_str(name, value, err, nominal):
    if not _is_constrained(err):
        return f"{name}={value:.2f} mm (UNCONSTRAINED -- edge not visible in this reference's footprint)"
    pull = (value - nominal) / err
    return f"{name}={value:.2f}+/-{err:.2f} mm (nominal {nominal:.1f} mm, pull {pull:+.1f} sigma)"


def _legend_value(name, value, err):
    """One legend line, e.g. "R = 10.07 +/- 0.02 mm" -- or, for a parameter the
    data don't constrain, no (meaningless) number at all."""
    if not _is_constrained(err):
        return f"{name}: edge outside beam"
    return f"{name} = {value:.2f} $\\pm$ {err:.2f} mm"


# Axis-label names for the references (filenames keep the short tags).
REF_DISPLAY = {"Tracker1": "Si tracker 1", "Tracker2": "Si tracker 2", "Hodoscope": "Hodoscope"}


def _plot_window(h_ref, xedges, yedges, center, half_span):
    """Plot crop: the fitted shape's own box (``center`` +/- ``half_span``),
    shrunk further to the bounding box of bins that entered the fit -- the
    reference detector's footprint often doesn't cover the whole box, and
    past the fitted bins there's only a sparse, mostly-empty fringe.
    """
    ix, iy = np.nonzero(h_ref >= MIN_REF_COUNT)
    window = []
    for c, h, edges, idx in ((center[0], half_span[0], xedges, ix), (center[1], half_span[1], yedges, iy)):
        lo, hi = max(c - h, edges[idx.min()]), min(c + h, edges[idx.max() + 1])
        window.append((lo, hi) if lo < hi else (c - h, c + h))
    return window


def _legend_loc(center, xlim, ylim):
    """The axes corner farthest from the fitted shape's center, so the
    legend sits over the eff=0 plateau rather than the shape's edge."""
    fx = (center[0] - xlim[0]) / (xlim[1] - xlim[0])
    fy = (center[1] - ylim[0]) / (ylim[1] - ylim[0])
    return f"{'upper' if fy < 0.5 else 'lower'} {'right' if fx < 0.5 else 'left'}"


def plot_fit_diagnostic(model, popt, eff, xedges, yedges, outline, fit_label, filename_stem,
                        rlabel, name, ref_label, center, xlim, ylim):
    """Save the data, fit and residual (data - fit) maps as three separate
    plots, ``<filename_stem>_{data,model,residual}.png``, each overlaid with
    the fitted shape's ``outline`` (an (xs, ys) closed curve) and a legend
    reading ``fit_label`` in the corner farthest from the shape (``_legend_loc``).
    ``name`` (e.g. "Veto") labels the colorbar. Same CaloX style and MAP_CMAP
    scale as the other 2D maps.
    """
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    residual = eff - model((X, Y), *popt)

    fine_xedges = np.linspace(*xlim, max(2, int(np.ceil((xlim[1] - xlim[0]) / MODEL_PLOT_BIN_MM))) + 1)
    fine_yedges = np.linspace(*ylim, max(2, int(np.ceil((ylim[1] - ylim[0]) / MODEL_PLOT_BIN_MM))) + 1)
    FX, FY = np.meshgrid(0.5 * (fine_xedges[:-1] + fine_xedges[1:]),
                         0.5 * (fine_yedges[:-1] + fine_yedges[1:]), indexing="ij")
    model_eff = model((FX, FY), *popt)

    panels = [("data", eff, xedges, yedges, f"{name} efficiency (data)", (0, 1)),
              ("model", model_eff, fine_xedges, fine_yedges, f"{name} efficiency (fit)", (0, 1)),
              ("residual", residual, xedges, yedges, "Data $-$ Fit", (-0.3, 0.3))]

    # Figure height scaled to the window's aspect so that, under
    # set_aspect("equal"), the axes keep the same width for any window --
    # otherwise a tall window narrows the axes and the CaloX header runs
    # into the run label. bbox_inches="tight" trims whatever's left over.
    figsize = (10, 10 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    plt.style.use(mh.style.ROOT)
    for suffix, values, xe, ye, zlabel, (vmin, vmax) in panels:
        fig, ax = plt.subplots(figsize=figsize)
        # cbarextend=False: hist2dplot's default of widening the figure for
        # the colorbar collapses the axes once set_aspect("equal") is applied.
        cb = mh.hist2dplot(values, xe, ye, ax=ax, cmap=MAP_CMAP, norm=Normalize(vmin, vmax),
                           flow=None, cbarextend=False, rasterized=True)
        cb.cbar.set_label(zlabel, loc="top")
        ax.set_facecolor(NO_DATA_COLOR)
        ax.plot(*outline, color="black", lw=2.5, label=fit_label)
        ax.legend(loc=_legend_loc(center, xlim, ylim), fontsize=18, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.9)
        mh.label.exp_label(ax=ax, exp="CaloX", text="Preliminary", rlabel=rlabel, data=True)
        ax.set_xlabel(f"{REF_DISPLAY.get(ref_label, ref_label)} X [mm]", loc="right")
        ax.set_ylabel(f"{REF_DISPLAY.get(ref_label, ref_label)} Y [mm]", loc="top")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        filename = f"{filename_stem}_{suffix}.png"
        plt.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    print(f"  Diagnostic plots saved to {filename_stem}_{{data,model,residual}}.png")


def report_veto(ref_label, result, output_dir, run_id):
    if result is None:
        print(f"  {ref_label} veto: no data, skipping")
        return
    popt, perr, maps = result
    x0, y0, r, sigma, eff0 = popt
    _, _, r_err, sigma_err, _ = perr
    print(f"  {ref_label} veto: {_param_str('R', r, r_err, VETO_RADIUS_MM)}, "
          f"sigma_smear = {sigma:.2f} +/- {sigma_err:.2f} mm, "
          f"center = ({x0:.1f}, {y0:.1f}) mm, plateau eff = {eff0:.3f}")
    eff, h_ref, xedges, yedges = maps
    tag = ref_label.lower().replace(" ", "_")
    filename_stem = os.path.join(output_dir, f"veto_fit_{tag}")
    t = np.linspace(0, 2 * np.pi, 400)
    outline = (x0 + r * np.cos(t), y0 + r * np.sin(t))
    fit_label = f"Fit: {_legend_value('R', r, r_err)}\n" + _legend_value("$\\sigma$", sigma, sigma_err)
    half_span = r + FIT_PLOT_MARGIN_MM
    xlim, ylim = _plot_window(h_ref, xedges, yedges, (x0, y0), (half_span, half_span))
    plot_fit_diagnostic(erf_disk, popt, eff, xedges, yedges, outline, fit_label, filename_stem,
                        run_header_label(run_id), "Veto", ref_label, (x0, y0), xlim, ylim)


def _report_counter(name, nominal_half_side, result, ref_label, output_dir, run_id):
    if result is None:
        print(f"  {ref_label} {name}: no data, skipping")
        return
    popt, perr, maps = result
    x0, y0, hx, hy, sigmax, sigmay, eff0 = popt
    _, _, hx_err, hy_err, sigmax_err, sigmay_err, _ = perr
    side_x, side_y = 2 * hx, 2 * hy
    side_x_err, side_y_err = 2 * hx_err, 2 * hy_err
    nominal_side = 2 * nominal_half_side
    print(f"  {ref_label} {name}: "
          f"{_param_str('side_x', side_x, side_x_err, nominal_side)}, "
          f"{_param_str('side_y', side_y, side_y_err, nominal_side)}, "
          f"sigma_smear = ({sigmax:.2f}, {sigmay:.2f}) mm, center = ({x0:.1f}, {y0:.1f}) mm, "
          f"plateau eff = {eff0:.3f}")
    eff, h_ref, xedges, yedges = maps
    tag = ref_label.lower().replace(" ", "_")
    slug = name.replace(" cm", "cm").replace(" ", "_").lower()  # "1 cm counter" -> 1cm_counter_fit_*.png
    filename_stem = os.path.join(output_dir, f"{slug}_fit_{tag}")
    outline = (x0 + hx * np.array([-1, 1, 1, -1, -1]), y0 + hy * np.array([-1, -1, 1, 1, -1]))
    fit_label = (f"Fit: {_legend_value('X side', side_x, side_x_err)}\n"
                 f"{_legend_value('Y side', side_y, side_y_err)}")
    xlim, ylim = _plot_window(h_ref, xedges, yedges, (x0, y0),
                              (hx + FIT_PLOT_MARGIN_MM, hy + FIT_PLOT_MARGIN_MM))
    plot_fit_diagnostic(erf_box, popt, eff, xedges, yedges, outline, fit_label, filename_stem,
                        run_header_label(run_id), name[0].upper() + name[1:], ref_label, (x0, y0), xlim, ylim)


def report_counter_1cm(ref_label, result, output_dir, run_id):
    _report_counter("1 cm counter", COUNTER_1CM_HALF_SIDE_MM, result, ref_label, output_dir, run_id)


def report_counter_3cm(ref_label, result, output_dir, run_id):
    _report_counter("3 cm counter", COUNTER_3CM_HALF_SIDE_MM, result, ref_label, output_dir, run_id)


def process_run(run_id):
    output_dir = ensure_output_dir(os.path.join("tracker_shapefit", str(run_id)))

    filepath = get_run_filepath(run_id)
    veto_branch, mcp1_branch, mcp2_branch = get_branch_names(run_id)
    # get_counter_branch_names already raises ValueError for a run before the
    # 1cm/3cm counters existed in the DAQ (< COUNTER_1CM_3CM_FIRST_RUN) --
    # catch that instead of letting it crash the whole run's veto/hodoscope
    # fit, which doesn't depend on the counters at all.
    try:
        one_cm_branch, three_cm_branch = get_counter_branch_names(run_id)
        has_counters = True
    except ValueError as e:
        print(f"  {e}")
        has_counters = False

    with uproot.open(filepath) as f:
        tree = f["EventTree"]
        trigger_n = tree["trigger_n"].array(library="np")
        root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
        veto_wf = np.stack(tree[veto_branch].array(library="np"))
        hg_x = np.stack(tree[X_HG_BRANCH].array(library="np"))[:, X_MAPPING]
        hg_y = np.stack(tree[Y_HG_BRANCH].array(library="np"))[:, Y_MAPPING]
        if has_counters:
            one_cm_wf = np.stack(tree[one_cm_branch].array(library="np"))
            three_cm_wf = np.stack(tree[three_cm_branch].array(library="np"))

    veto_sel = passes_veto(veto_wf, threshold=VETO_THRESHOLD)
    one_cm_hit = counter_1cm_hit_mask(one_cm_wf) if has_counters else None
    three_cm_hit = counter_3cm_hit_mask(three_cm_wf) if has_counters else None
    xh, yh, good_hodo = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH, run_id=run_id)

    si_data = load_tracker_run(run_id)
    # run_id here is what makes build_aligned_tracker_branches add the
    # calibrated tracker_x{1,2}_mm/tracker_y{1,2}_mm branches (see its and
    # utils.tracker.calibrate_tracker_positions's docstrings) -- this is the
    # reconstruction-time calibration step, shared with every other caller
    # of build_aligned_tracker_branches, rather than logic private to this
    # script.
    tracker_branches, match_frac = build_aligned_tracker_branches(si_data, trigger_n, root_tstamp, run_id=run_id)
    x1, y1 = tracker_branches["tracker_x1"], tracker_branches["tracker_y1"]
    x2, y2 = tracker_branches["tracker_x2"], tracker_branches["tracker_y2"]
    x1_mm, y1_mm = tracker_branches["tracker_x1_mm"], tracker_branches["tracker_y1_mm"]
    x2_mm, y2_mm = tracker_branches["tracker_x2_mm"], tracker_branches["tracker_y2_mm"]
    counter_summary = (f"1cm_hits={one_cm_hit.sum()}, 3cm_hits={three_cm_hit.sum()}" if has_counters
                       else "1cm/3cm counters not present in this run")
    print(f"  [{run_id}] tracker match_frac={match_frac:.4%}, "
          f"veto_hits={veto_sel.sum()}, {counter_summary}, hodo_good={good_hodo.sum()}")

    # Three independent reference "rulers" -- each detector's own (x, y) in
    # mm, and the mask of events where that detector reports a real
    # position. The Tracker1/Tracker2 positions above are already
    # calibrated into the hodoscope's frame.

    references = {
        "Tracker1": (x1_mm, y1_mm, (x1 != SENTINEL_X1) & (y1 != SENTINEL_Y1)),
        "Tracker2": (x2_mm, y2_mm, (x2 != SENTINEL_X2) & (y2 != SENTINEL_Y2)),
        "Hodoscope": (xh, yh, good_hodo),
    }

    beam_label = get_beam_label(run_id)
    print(f"\nFitting shapes ({beam_label})...")
    for ref_label, (x_r, y_r, ref) in references.items():
        if not ref.any():
            print(f"  {ref_label}: no reference hits, skipping")
            continue
        x_ref, y_ref = x_r[ref], y_r[ref]
        hodo = ref_label == "Hodoscope"

        sel = ref & veto_sel
        result = fit_veto(x_ref, y_ref, x_r[sel], y_r[sel], hodo=hodo)
        report_veto(ref_label, result, output_dir, run_id)

        if not has_counters:
            print(f"  {ref_label} 1cm/3cm counter: no counter data for run {run_id} "
                  f"(< {COUNTER_1CM_3CM_FIRST_RUN}), skipping")
            continue

        sel = ref & one_cm_hit
        result = fit_counter_1cm(x_ref, y_ref, x_r[sel], y_r[sel], hodo=hodo)
        report_counter_1cm(ref_label, result, output_dir, run_id)

        sel = ref & three_cm_hit
        result = fit_counter_3cm(x_ref, y_ref, x_r[sel], y_r[sel], hodo=hodo)
        report_counter_3cm(ref_label, result, output_dir, run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True,
                        help="Run ID with tracker data. 1cm/3cm counter fits are skipped "
                             f"for runs before {COUNTER_1CM_3CM_FIRST_RUN}.")
    args = parser.parse_args()

    print(f"Loading run {args.run}...")
    process_run(args.run)
    print("Done.")


if __name__ == "__main__":
    main()
