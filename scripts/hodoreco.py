import uproot
import mplhep as mh
import matplotlib.pyplot as plt
import numpy as np

import os 
import sys
import argparse 
import json 

from scripts.sitrackreco import OUTPUTDIR
from utils.hodo import reconstruct_hodoscope
from utils.constants import (HG_THRESHOLD, X_MAPPING, Y_MAPPING, PITCH)
from utils.plotting import get_runtype, plot_profile
from utils.data import load_run_list

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

            results = {
                "xh": xh[mask],
                "yh": yh[mask],
                "n_events": mask.sum(),
                "hgx": hg_x[mask],
                "hgy": hg_y[mask]
            }
            return results
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
        return None
    
def plot_energy_histogram(hg_x, hg_y, run_id, OUTPUTDIR=OUTPUTDIR, fname="HG_dist"):
    runtype = get_runtype(run_id)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 10))
    h1 = mh.histplot(np.histogram(hg_x.flatten(), bins=100), ax=ax, label="X Hodoscope Energy", color="blue")
    h2 = mh.histplot(np.histogram(hg_y.flatten(), bins=100), ax=ax, label="Y Hodoscope Energy", color="red")
    mh.label.exp_label(ax=ax, exp="CaloX", text=runtype, rlabel="Hodoscope Energy", data=True)
    ax.set_xlabel("Hodoscope Energy [ADC]")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(os.path.join(OUTPUTDIR, f"{fname}_{run_id}.png"), dpi=300)
    print("Profile Plot Saved " + os.path.join(OUTPUTDIR, f"{fname}_{run_id}.png"))
    plt.close()

def main(): 
    parser = argparse.ArgumentParser(description="Hodoscope Reconstruction")
    parser.add_argument("--run", type=str, required=True, help="Run ID to process")
    args = parser.parse_args()

    OUTPUTDIR = f"/lustre/work/colnunn/detreco/output/hodoreco/{args.run}"
    os.makedirs(OUTPUTDIR, exist_ok=True)

    run_list = load_run_list()
    if args.run not in run_list:
        print(f"Run ID {args.run} not found in run list.")
        sys.exit(1)

    result = hodoreco((args.run, run_list[args.run]["file"]))
    if result is not None:
        xh = result["xh"]
        yh = result["yh"]
        n_events = result["n_events"]
        hgx = result["hgx"]
        hgy = result["hgy"]
        plot_profile(xh, yh, args.run, OUTPUTDIR=OUTPUTDIR)
        plot_energy_histogram(hgx, hgy, args.run, OUTPUTDIR=OUTPUTDIR)
        # plot_hit_histogram(hgx, hgy, args.run, OUTPUTDIR=OUTPUTDIR)
        print(f"Processed {n_events} events for run {args.run}.")
    else:
        print(f"Failed to process run {args.run}.")

if __name__ == "__main__":
    main()