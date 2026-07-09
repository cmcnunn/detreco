import uproot
import mplhep as mh
import matplotlib.pyplot as plt
import numpy as np

import os 
import sys
import argparse 
import json 

from utils.hodo import reconstruct_hodoscope
from utils.constants import (HG_THRESHOLD, X_MAPPING, Y_MAPPING, PITCH)
from utils.plotting import get_runtype
from utils.data import load_run_list

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/hodoreco_output"
os.makedirs(OUTPUTDIR, exist_ok=True)

Y_HG = "FERS_Board0_energyHG"
X_HG = "FERS_Board1_energyHG"

def hodoreco(run_data):
    run_id, file_path = run_data

    try: 
        with uproot.open(file_path) as f:
            tree = f["EventTree"]
            hg_x = np.stack(tree[X_HG].array(library="np"))[:, X_MAPPING]
            hg_y = np.stack(tree[Y_HG].array(library="np"))[:, Y_MAPPING]
            xh, yh, mask = reconstruct_hodoscope(hg_x, hg_y, threshold=HG_THRESHOLD, pitch=PITCH)
            return xh[mask], yh[mask], mask.sum()
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
        return None
    
def plot_hodoprofile(xh, yh, run_id, runtype=""):
    if runtype == "":
        runtype = get_runtype(run_id)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(12, 12))
    H = np.histogram2d(xh, yh, bins=64)
    cb = mh.hist2dplot(*H, ax=ax, cmin=0)
    cb.cbar.set_label("Events", loc='top')
    mh.cms.label(ax=ax, exp="CaloX", text=runtype, rlabel=r"Hodoscope Profile 2D", data=True)
    ax.set_xlabel("Hodo X [cm]", loc='right')
    ax.set_ylabel("Hodo Y [cm]", loc='top')
    plt.savefig(os.path.join(OUTPUTDIR, f"hodo_profile2d_{run_id}.png"), dpi=300)
    print("Profile Plot Saved " + os.path.join(OUTPUTDIR, f"hodo_profile2d_{run_id}.png"))

def main(): 
    parser = argparse.ArgumentParser(description="Hodoscope Reconstruction")
    parser.add_argument("--run", type=str, required=True, help="Run ID to process")
    args = parser.parse_args()

    run_list = load_run_list()
    if args.run not in run_list:
        print(f"Run ID {args.run} not found in run list.")
        sys.exit(1)

    result = hodoreco((args.run, run_list[args.run]))
    if result is not None:
        xh, yh, n_events = result
        plot_hodoprofile(xh, yh, args.run)
        print(f"Processed {n_events} events for run {args.run}.")
    else:
        print(f"Failed to process run {args.run}.")

if __name__ == "__main__":
    main()