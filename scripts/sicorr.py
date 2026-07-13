"""Check whether si-tracker station 1 and station 2 misses are correlated, for one run.

If a shared upstream cause (e.g. a DAQ/readout hiccup, a missing tracker
row) drops both stations on the same event, misses will cluster together
far more than chance. If each station is independently inefficient,
P(Si2 miss | Si1 miss) should land close to the marginal P(Si2 miss).

Reports the 2x2 contingency table (over every reference-selected event in
the run), the two conditional miss probabilities, and the phi coefficient
(binary correlation, [-1, 1]).

Usage:
    python scripts/sicorr.py --run <run_id>
"""

import argparse
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from scripts.sieff import load_si_hit_masks
from utils.fit_funcs import line
from utils.io import ensure_output_dir
from utils.plotting import get_runtype

OUTPUT_DIR = ensure_output_dir("sicorr")
_FIT_OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]


def report_correlation(hit1, hit2):
    n = len(hit1)
    n11 = int(np.sum(hit1 & hit2))    # both hit
    n10 = int(np.sum(hit1 & ~hit2))   # si1 hit, si2 miss
    n01 = int(np.sum(~hit1 & hit2))   # si1 miss, si2 hit
    n00 = int(np.sum(~hit1 & ~hit2))  # both miss

    n_si1_hit, n_si1_miss = n11 + n10, n01 + n00
    n_si2_hit, n_si2_miss = n11 + n01, n10 + n00

    p_miss1 = n_si1_miss / n
    p_miss2 = n_si2_miss / n
    p_miss2_given_miss1 = n00 / n_si1_miss if n_si1_miss else float("nan")
    p_miss1_given_miss2 = n00 / n_si2_miss if n_si2_miss else float("nan")

    denom = np.sqrt(n_si1_hit * n_si1_miss * n_si2_hit * n_si2_miss)
    phi = (n11 * n00 - n10 * n01) / denom if denom else float("nan")

    print(f"\nn_ref_events = {n}")
    print("Contingency table:")
    print(f"{'':<12}{'Si2 hit':>12}{'Si2 miss':>12}")
    print(f"{'Si1 hit':<12}{n11:>12d}{n10:>12d}")
    print(f"{'Si1 miss':<12}{n01:>12d}{n00:>12d}")
    print()
    print(f"P(Si1 miss)            = {p_miss1:.4f}")
    print(f"P(Si2 miss)            = {p_miss2:.4f}  <- marginal (baseline) rate")
    if p_miss2:
        print(f"P(Si2 miss | Si1 miss) = {p_miss2_given_miss1:.4f}  "
              f"({p_miss2_given_miss1 / p_miss2:.2f}x the marginal rate)")
    if p_miss1:
        print(f"P(Si1 miss | Si2 miss) = {p_miss1_given_miss2:.4f}  "
              f"({p_miss1_given_miss2 / p_miss1:.2f}x the marginal rate)")
    print(f"phi coefficient (binary correlation, hit=1/miss=0) = {phi:.4f}")
    print("  (0 = independent stations; toward 1 = misses/hits cluster on the same events,")
    print("   consistent with a shared upstream cause rather than two independent inefficiencies)")


def plot_hit_correlation(hit1, hit2, run_id, filename):
    """2D hit/miss heatmap with a straight-line fit through the raw (0/1) events.

    ``profile_mode``/``draw_fit`` (used elsewhere for spatial correlations)
    bin x into 64 slices and fit the per-slice peak -- that degenerates
    when x only takes two values (0 and 1), so this fits the raw events
    directly instead. For 0/1 data that's an ordinary least-squares fit of
    Si2 on Si1, and its Pearson r is mathematically identical to the phi
    coefficient reported by ``report_correlation``.
    """
    x, y = hit1.astype(float), hit2.astype(float)

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 10))
    edges = np.array([-0.5, 0.5, 1.5])
    H = np.histogram2d(x, y, bins=[edges, edges])
    # cmin/cmax can't be passed alongside norm (pcolormesh forwards both as
    # vmin/vmax and errors on the conflict); LogNorm needs vmin > 0 since
    # log(0) is undefined -- fine here as long as no cell is exactly 0.
    cb = mh.hist2dplot(*H, ax=ax, norm=LogNorm(vmin=1))
    cb.cbar.set_label("Events", loc="top")

    (m, b), cov = curve_fit(line, x, y)
    m_err, b_err = np.sqrt(np.diag(cov))
    r = np.corrcoef(x, y)[0, 1]
    xs = np.array([0.0, 1.0])
    ax.plot(xs, line(xs, m, b), color="white", lw=2, path_effects=_FIT_OUTLINE)
    ax.text(0.5, 0.5, f"y = ({m:.3f} $\\pm$ {m_err:.3f})x + ({b:.3f} $\\pm$ {b_err:.3f})\n$r$ = {r:.4f}",
            transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=20, path_effects=_FIT_OUTLINE)

    runtype = get_runtype(run_id)
    mh.cms.label(ax=ax, exp="CaloX", text=runtype, rlabel=f"Si1 vs Si2 Hits — run {run_id}", data=True)
    ax.set_xlabel("Si Tracker 1 (0=miss, 1=hit)", loc="right")
    ax.set_ylabel("Si Tracker 2 (0=miss, 1=hit)", loc="top")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(filename)
    print("Hit-correlation plot saved " + filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, required=True, help="Run ID to process")
    args = parser.parse_args()

    try:
        hit1, hit2 = load_si_hit_masks(args.run)
    except Exception as e:
        print(f"Error processing run {args.run}: {e}")
        return

    print(f"Run {args.run}")
    report_correlation(hit1, hit2)

    sub_outdir = os.path.join(OUTPUT_DIR, args.run)
    os.makedirs(sub_outdir, exist_ok=True)
    plot_hit_correlation(hit1, hit2, args.run, os.path.join(sub_outdir, f"hitcorr_{args.run}.png"))


if __name__ == "__main__":
    main()
