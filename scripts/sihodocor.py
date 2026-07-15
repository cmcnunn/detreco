import os
import argparse

import numpy as np
import uproot
import mplhep as mh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.data import get_run_filepath
from utils.tracker import load_tracker_run, station1_hit_mask, station2_hit_mask, align_tracker_to_root_by_timestamp
from utils.plotting import get_beam_label, draw_fit, _hist_edges
from utils.constants import X_MAPPING, Y_MAPPING
from utils.hodo import reconstruct_hodoscope
from utils.selectors import get_branch_names, passes_veto, get_counter_branch_names, counter_1cm_hit_mask, counter_3cm_hit_mask

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/sihodocor"
os.makedirs(OUTPUTDIR, exist_ok=True)

# White-on-black-outline so the fit line/annotation stay legible over every
# viridis bin color, from the dark low-count purple to the bright yellow peak.

def plot_sihodocor(xh, yh, x1, y1, run_id, trackern, selection="", runtype="", OUTPUTDIR=OUTPUTDIR):
    if runtype == "":
        runtype = get_beam_label(run_id)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(xh, x1, bins=[_hist_edges(xh), _hist_edges(x1)])
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    tag = f"Run {run_id}" + (f" -- {selection}" if selection else "")
    draw_fit(ax, xh, x1, tag=tag)
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=f"HodoX vs Tracker{trackern}", data=True)
    ax.set_xlabel("Hodo X [mm]", loc='right')
    ax.set_ylabel("Silicon Tracker X [mm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_x_{run_id}_{selection}.png"), dpi=300)
    print("Hodo vs Tracker Plot Saved " + os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_x_{run_id}_{selection}.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(yh, y1, bins=[_hist_edges(yh), _hist_edges(y1)])
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    draw_fit(ax, yh, y1, tag=tag)
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel=f"HodoY vs Tracker{trackern}", data=True)
    ax.set_xlabel("Hodo Y [mm]", loc='right')
    ax.set_ylabel("Silicon Tracker Y [mm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_y_{run_id}_{selection}.png"), dpi=300)
    print("Hodo vs Tracker Plot Saved " + os.path.join(OUTPUTDIR, f"sihodocor_{trackern}_y_{run_id}_{selection}.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct hodoscope hits from tracker data."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()
    run_output_dir = os.path.join(OUTPUTDIR, args.run)
    os.makedirs(run_output_dir, exist_ok=True)
    filepath = get_run_filepath(args.run)
    si_data = load_tracker_run(args.run)
    mask1, mask2 = station1_hit_mask(si_data), station2_hit_mask(si_data)
    veto_branch, _, _ = get_branch_names(args.run)
    if int(args.run) >= 1825:
        one_cm_branch, three_cm_branch = get_counter_branch_names(args.run)
    try:
        with uproot.open(filepath) as f:
            tree = f["EventTree"]
            trigger_n = tree["trigger_n"].array(library="np")
            root_tstamp = tree["FERS_Board1_tstamp_us"].array(library="np")
            hg_x = np.stack(tree["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
            hg_y = np.stack(tree["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
            xh, yh, maskh = reconstruct_hodoscope(hg_x, hg_y, threshold=4000, pitch=0.6)
            veto_wf = np.stack(tree[veto_branch].array(library="np"))
            maskv = passes_veto(veto_wf)
            if int(args.run) >= 1825:
                one_cm_wf = np.stack(tree[one_cm_branch].array(library="np"))
                three_cm_wf = np.stack(tree[three_cm_branch].array(library="np"))
                one_cm_hit = counter_1cm_hit_mask(one_cm_wf)
                three_cm_hit = counter_3cm_hit_mask(three_cm_wf)
    except Exception as e:
        print(f"Error processing ROOT file {args.run}: {e}")
        return

    # Tracker and ROOT arrays don't share an index space yet -- align_tracker_to_root_by_timestamp
    # finds per-spill-segment offsets between si_data's run_event_nr and DREAM's trigger_n
    # (using real hardware timestamps to locate spill boundaries), and gives back the row
    # indices needed to line the two up.
    tracker_mask, root_idx, offsets, match_frac = align_tracker_to_root_by_timestamp(si_data, trigger_n, root_tstamp)
    print(f"Aligned {tracker_mask.sum()}/{len(si_data)} tracker events "
          f"({len(offsets)} segments, match_frac={match_frac:.4%})")

    si_aligned = si_data[tracker_mask]
    xh_aligned, yh_aligned, maskh_aligned = xh[root_idx], yh[root_idx], maskh[root_idx]
    maskv_aligned = maskv[root_idx]
    if int(args.run) >= 1825:
        one_cm_hit_aligned = one_cm_hit[root_idx]
        three_cm_hit_aligned = three_cm_hit[root_idx]

    # Now every array below is in the same, aligned order -- combine the
    # per-station tracker hit masks (re-sliced to the aligned subset) with
    # the hodoscope goodness mask and the veto-counter mask. The veto plane
    # fires on beam-halo particles outside the region of interest, which is
    # exactly the wide low-density background diluting the tracker/hodoscope
    # correlation, so this should cut most of that noise.
    mask = mask1[tracker_mask] & mask2[tracker_mask] & maskh_aligned
    print(f"{mask.sum()}/{len(si_aligned)} aligned events pass all masks (incl. veto)")
    plot_list = ["", "Veto_Selection", "1_CM_Counter", "3_CM_Counter"]
    for plot in plot_list:
        if plot == "":
            mask_plot = mask
        elif plot == "Veto_Selection":
            mask_plot = mask & maskv_aligned
        elif plot == "1_CM_Counter" and int(args.run) >= 1825:
            mask_plot = mask & one_cm_hit_aligned
        elif plot == "3_CM_Counter" and int(args.run) >= 1825:
            mask_plot = mask & three_cm_hit_aligned
        else:
            continue
        #convert to mm 
        x1, y1 = 10*si_aligned["x1"][mask_plot], 10*si_aligned["y1"][mask_plot]
        x2, y2 = 10*si_aligned["x2"][mask_plot], 10*si_aligned["y2"][mask_plot]
        xh_good, yh_good = xh_aligned[mask_plot], yh_aligned[mask_plot]

        plot_sihodocor(xh_good, yh_good, x1, y1, args.run, trackern="1", selection=plot, OUTPUTDIR=run_output_dir)
        plot_sihodocor(xh_good, yh_good, x2, y2, args.run, trackern="2", selection=plot, OUTPUTDIR=run_output_dir)
    

if __name__ == "__main__":
    main()
