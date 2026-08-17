import os 
import numpy as np
import uproot
import argparse

import matplotlib.pyplot as plt
import mplhep as mh

from utils.tracker import (
    SENTINEL_X1,
    SENTINEL_X2,
    SENTINEL_Y1,
    SENTINEL_Y2,
    build_aligned_tracker_branches,
    load_tracker_run,
)
from utils.hodo import good_hodo_mask, reconstruct_hodoscope 
from utils.constants import X_MAPPING, Y_MAPPING
from utils.data import get_run_filepath, get_run_beam
from utils.plotting import get_beam_label
from utils.energy import load_energy_data


def main():
    parser = argparse.ArgumentParser(
    description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()
    try: 
        with uproot.open(get_run_filepath(args.run)) as f:
            t = f["EventTree"]
            # Load Hodoscope Energy
            HGx = np.stack(t["FERS_Board1_energyHG"].array(library="np"))
            HGy = np.stack(t["FERS_Board2_energyHG"].array(library="np"))
        hx, hy, good_hodo = reconstruct_hodoscope(HGx, HGy)
        #Load Tracker data 
        tracker_data = load_tracker_run(args.run)
        #Load energy data 
        total_sci_energy, total_cer_energy = load_energy_data(args.run, calib_data=False)

    except Exception as e:
        print(f"Error processing run {args.run}: {e}")
        return
if __name__ == "__main__":
    main()