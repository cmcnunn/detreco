import uproot
import mplhep as mh
import matplotlib.pyplot as plt
import numpy as np

import os 
import sys
import argparse 
import json 

from utils.tracker import load_tracker_run, station1_hit_mask, station2_hit_mask
from utils.plotting import get_beam_label, plot_profile

global OUTPUTDIR
OUTPUTDIR = "/lustre/work/colnunn/detreco/output/sitrackreco"
os.makedirs(OUTPUTDIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Scilicon Strip Detector Reconstruction")
    parser.add_argument("--run", type=str, required=True, help="Run ID to process")
    args = parser.parse_args()

    data = load_tracker_run(args.run)
    mask1, mask2 = station1_hit_mask(data), station2_hit_mask(data)
    x1, y1 = data["x1"][mask1], data["y1"][mask1]
    x2, y2 = data["x2"][mask2], data["y2"][mask2]
    runtype = get_beam_label(args.run)
    plot_profile(x1, y1, args.run, runtype=runtype, OUTPUTDIR=OUTPUTDIR, label=f"{args.run} Si Tracker 1", fname="sitrack1_profile2d")
    plot_profile(x2, y2, args.run, runtype=runtype, OUTPUTDIR=OUTPUTDIR, label=f"{args.run} Si Tracker 2", fname="sitrack2_profile2d")
if __name__ == "__main__":
    main()