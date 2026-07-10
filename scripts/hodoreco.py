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
from utils.plotting import get_runtype, plot_profile
from utils.data import load_run_list

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/hodoreco"
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
        plot_profile(xh, yh, args.run, OUTPUTDIR=OUTPUTDIR)
        print(f"Processed {n_events} events for run {args.run}.")
    else:
        print(f"Failed to process run {args.run}.")

if __name__ == "__main__":
    main()