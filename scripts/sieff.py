import os 
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import mplhep as mh
import uproot

from utils.data import get_run_filepath
from utils.tracker import load_tracker_run, station1_hit_mask, station2_hit_mask, align_tracker_to_root
from utils.plotting import get_runtype, plot_effhist2d
from utils.selectors import passes_veto, get_branch_names
from utils.hodo import reconstruct_hodoscope
from utils.constants import HG_THRESHOLD, PITCH, VETO_THRESHOLD, X_MAPPING, Y_MAPPING

global OUTPUTDIR 
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/sieff"
os.makedirs(OUTPUTDIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()

    filepath = get_run_filepath(args.run)
    veto, _, _ = get_branch_names(args.run)
    try:
        with uproot.open(filepath) as f:
            tree = f["EventTree"]
            trigger_n = tree["trigger_n"].array(library="np")
            xh = np.stack(tree["FERS_Board1_energyHG"].array(library="np"))[:, X_MAPPING]
            yh = np.stack(tree["FERS_Board0_energyHG"].array(library="np"))[:, Y_MAPPING]
            veto_wf = np.stack(tree[veto].array(library="np"))
    except Exception as e:
        print(f"Error loading ROOT file {args.run}: {e}")
        return
    
    mask_v = passes_veto(veto_wf)
    xh, yh, good_hodo = reconstruct_hodoscope(xh, yh)
    si_data = load_tracker_run(args.run)
    mask1, mask2 = station1_hit_mask(si_data), station2_hit_mask(si_data)

    tracker_mask, root_idx, offset, match_frac = align_tracker_to_root(si_data, trigger_n)
    mask1_aligned, mask2_aligned = mask1[tracker_mask], mask2[tracker_mask]
    xh_aligned, yh_aligned, maskh_aligned = xh[root_idx], yh[root_idx], good_hodo[root_idx]
    maskv_aligned = mask_v[root_idx]

    ref = maskh_aligned & maskv_aligned
    xh_ref, yh_ref = xh_aligned[ref], yh_aligned[ref]
    xh_sel1, yh_sel1 = xh_aligned[ref & mask1_aligned], yh_aligned[ref & mask1_aligned]
    xh_sel2, yh_sel2 = xh_aligned[ref & mask2_aligned], yh_aligned[ref & mask2_aligned]

    runtype = get_runtype(args.run)
    sub_outdir = os.path.join(OUTPUTDIR, args.run)
    os.makedirs(sub_outdir, exist_ok=True)

    plot_effhist2d(xh_ref, yh_ref, xh_sel1, yh_sel1, 64, "Hodo X [mm]", "Hodo Y [mm]",
                   "Si Tracker 1 Efficiency", os.path.join(sub_outdir, f"si1_effmap_{args.run}.png"),
                   runtype=runtype)
    plot_effhist2d(xh_ref, yh_ref, xh_sel2, yh_sel2, 64, "Hodo X [mm]", "Hodo Y [mm]",
                   "Si Tracker 2 Efficiency", os.path.join(sub_outdir, f"si2_effmap_{args.run}.png"),
                   runtype=runtype)

if __name__ == "__main__":
    main()