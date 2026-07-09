"""Hodoscope-referenced 2D efficiency maps for MCP1, MCP2, and the Veto."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import HG_THRESHOLD, X_MAPPING, Y_MAPPING, PITCH, VETO_THRESHOLD
from utils.hodo import reconstruct_hodoscope
from utils.io import ensure_output_dir

from utils.selectors import get_branch_names, get_mcp_pulse_window_ns, passes_veto
from utils.waveforms import subtract_baseline

# --- Branch names ---
Y_HG = "FERS_Board0_energyHG"
X_HG = "FERS_Board1_energyHG"

# --- Thresholds ---
LG_THRESHOLD = 229
DRS_THRESHOLD = -13.6

# --- MCP pulse finding (matches mcp_studies.py) ---
# Pulse window (ns) is run-dependent; see utils.selectors.get_mcp_pulse_window_ns.
MCP_NOISE_SAMPLES = 50
MCP_NOISE_MARGIN_ADC = 20
MCP_MIN_PULSE_FWHM_NS = 1.0

# --- Geometry ---
SAMPLE_NS = 0.2

OUTPUT_DIR = ensure_output_dir("effplots")


# ==============================
# Plotting helpers
# ==============================
def plot_effhist2d(x_ref, y_ref, x_sel, y_sel, bins, xlabel, ylabel, title, filename):
    x_bins = np.linspace(-PITCH * 32, PITCH * 32, bins)
    y_bins = np.linspace(-PITCH * 32, PITCH * 32, bins)

    h_ref, xedges, yedges = np.histogram2d(x_ref, y_ref, bins=[x_bins, y_bins])
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[x_bins, y_bins])

    eff = np.divide(h_sel, h_ref, out=np.zeros_like(h_sel, dtype=float), where=h_ref > 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    pc = mh.hist2dplot(eff, xedges, yedges, ax=ax, cmin=0, cmax=1, rasterized=True)
    cb = pc.cbar
    if cb:
        cb.set_label("Efficiency", loc='top')
    mh.label.exp_label(exp="CaloX", data=True, rlabel=title, ax=ax)
    ax.set_xlabel(xlabel, loc='right')
    ax.set_ylabel(ylabel, loc='top')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_effhist1d(x_ref, y_ref, x_sel, y_sel, xlabel, ylabel, title, filename):
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


def _mcp_hit_mask(mcp_bs, pulse_window_ns):
    """True for events with a pulse minimum below dynamic threshold inside the window."""
    w_start = int(pulse_window_ns[0] / SAMPLE_NS)
    w_end = int(pulse_window_ns[1] / SAMPLE_NS)
    noise_avg = np.mean(np.abs(mcp_bs[:, :MCP_NOISE_SAMPLES]), axis=1)
    thresh = -(noise_avg + MCP_NOISE_MARGIN_ADC)
    return mcp_bs[:, w_start:w_end + 1].min(axis=1) < thresh

def get_intrinsic_efficiency(x_ref, y_ref, x_sel, y_sel, return_uncertainty=True):
    """Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."""
    h_ref, xedges, yedges = np.histogram2d(x_ref, y_ref, bins=64)
    h_sel, _, _ = np.histogram2d(x_sel, y_sel, bins=[xedges, yedges])

    eff = np.divide(h_sel, h_ref, out=np.zeros_like(h_sel, dtype=float), where=h_ref > 0)
    geometric_mask = eff > 0.5
    eff = eff[geometric_mask] #mask to only consider bins where there are reference hits
    intrinsic_efficiency = np.mean(eff)  
    if return_uncertainty:
        # Calculate uncertainty using binomial statistics
        n_bins = np.sum(geometric_mask)
        if n_bins > 0:
            uncertainty = np.sqrt(intrinsic_efficiency * (1 - intrinsic_efficiency) / n_bins)
            return intrinsic_efficiency, uncertainty
        else:
            return intrinsic_efficiency, None
    else:
        return intrinsic_efficiency

def process_single_run(run_data):
    run_id, file_path = run_data
    VETO, MCP1, MCP2 = get_branch_names(run_id)
    pulse_window_ns = get_mcp_pulse_window_ns(run_id)
    try:
        with uproot.open(file_path) as f:
            tree = f["EventTree"]
            hg_x = np.stack(tree[X_HG].array(library="np"))[:, X_MAPPING]
            hg_y = np.stack(tree[Y_HG].array(library="np"))[:, Y_MAPPING]
            veto = np.stack(tree[VETO].array(library="np"))
            mcp1 = subtract_baseline(np.stack(tree[MCP1].array(library="np")))
            mcp2 = subtract_baseline(np.stack(tree[MCP2].array(library="np")))

        xh, yh, good_hodo = reconstruct_hodoscope(
            hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH, method="mean",
        )

        veto_sel = passes_veto(veto, threshold=VETO_THRESHOLD)
        mcp1_hit = _mcp_hit_mask(mcp1, pulse_window_ns)
        mcp2_hit = _mcp_hit_mask(mcp2, pulse_window_ns)

        w_start = int(pulse_window_ns[0] / SAMPLE_NS)
        w_end = int(pulse_window_ns[1] / SAMPLE_NS)
        noise_avg = np.mean(np.abs(mcp1[:, :MCP_NOISE_SAMPLES]), axis=1)
        intrinsic_efficiency_mcp1, uncertainty_mcp1 = get_intrinsic_efficiency(xh[good_hodo], yh[good_hodo],
                                                                               xh[good_hodo & mcp1_hit], yh[good_hodo & mcp1_hit])
        intrinsic_efficiency_mcp2, uncertainty_mcp2 = get_intrinsic_efficiency(xh[good_hodo], yh[good_hodo],
                                                                               xh[good_hodo & mcp2_hit], yh[good_hodo & mcp2_hit])
        intrinsic_efficiency_veto, uncertainty_veto = get_intrinsic_efficiency(xh[good_hodo], yh[good_hodo],
                                                                               xh[good_hodo & veto_sel], yh[good_hodo & veto_sel])
        print(f"  [{run_id}] n_events={len(mcp1)}  "
              f"mcp1_hits={mcp1_hit.sum()}  mcp2_hits={mcp2_hit.sum()}  veto_hits={veto_sel.sum()}  "
              f"mean_thresh={-(noise_avg.mean() + MCP_NOISE_MARGIN_ADC):.1f}  "
              f"mean_window_min={mcp1[:, w_start:w_end+1].min(axis=1).mean():.1f} "
              f"intrinsic_eff_mcp1={intrinsic_efficiency_mcp1:.3f} ± {uncertainty_mcp1:.3f}  "
              f"intrinsic_eff_mcp2={intrinsic_efficiency_mcp2:.3f} ± {uncertainty_mcp2:.3f}  "
              f"intrinsic_eff_veto={intrinsic_efficiency_veto:.3f} ± {uncertainty_veto:.3f}")

        ref = good_hodo
        return (xh[ref], yh[ref],
                xh[ref & mcp1_hit], yh[ref & mcp1_hit],
                xh[ref & mcp2_hit], yh[ref & mcp2_hit],
                xh[ref & veto_sel], yh[ref & veto_sel])
    except Exception as e:
        print(f"Error in {run_id}: {e}")
        return [], [], [], [], [], [], [], []


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run ID to process (default: all runs)")
    args = parser.parse_args()

    try:
        with open("run_list.json", "r") as f:
            run_files = json.load(f)
    except Exception:
        with open("data/run_list.json", "r") as f:
            run_files = json.load(f)

    if args.run is not None:
        if args.run not in run_files:
            raise SystemExit(f"Run '{args.run}' not found in run_list.json")
        run_files = {args.run: run_files[args.run]}

    runs = list(run_files.items())
    runs_label = ", ".join(r[0] for r in runs)

    print(f"Processing {len(runs)} run(s) sequentially...")
    results = [process_single_run(run) for run in runs]

    X_ref = np.concatenate([r[0] for r in results])
    Y_ref = np.concatenate([r[1] for r in results])
    X1_sel = np.concatenate([r[2] for r in results])
    Y1_sel = np.concatenate([r[3] for r in results])
    X2_sel = np.concatenate([r[4] for r in results])
    Y2_sel = np.concatenate([r[5] for r in results])
    Xv = np.concatenate([r[6] for r in results])
    Yv = np.concatenate([r[7] for r in results])

    plt.style.use(mh.style.ROOT)
    plot_effhist2d(X_ref, Y_ref, X1_sel, Y1_sel, 64,
                   "X Position [mm]", "Y Position [mm]", f"MCP1 — {runs_label}",
                   os.path.join(OUTPUT_DIR, f"hodo_mcp1effmap_{runs_label}.pdf"))
    print(f"Efficiency map for MCP1 saved to {os.path.join(OUTPUT_DIR, f'hodo_mcp1effmap_{runs_label}.pdf')}")

    plot_effhist2d(X_ref, Y_ref, X2_sel, Y2_sel, 64,
                   "X Position [mm]", "Y Position [mm]", f"MCP2 — {runs_label}",
                   os.path.join(OUTPUT_DIR, f"hodo_mcp2effmap_{runs_label}.pdf"))
    print(f"Efficiency map for MCP2 saved to {os.path.join(OUTPUT_DIR, f'hodo_mcp2effmap_{runs_label}.pdf')}")

    plot_effhist2d(X_ref, Y_ref, Xv, Yv, 64,
                   "X Position [mm]", "Y Position [mm]", f"Veto — {runs_label}",
                   os.path.join(OUTPUT_DIR, f"hodo_vetoeffmap_{runs_label}.pdf"))
    print(f"Efficiency map for Veto saved to {os.path.join(OUTPUT_DIR, f'hodo_vetoeffmap_{runs_label}.pdf')}")

    print("Aggregation complete. Generating plots...")
    print("Done.")


if __name__ == "__main__":
    main()
