import numpy as np
import json 
import uproot

from .data import get_run_filepath

def check_file(run, DRS_START = 1820):
    if run <= DRS_START:
        return True
    else:
        return False


def load_configs():
    '''
    Load scintillator and Cherenkov channel configurations from JSON files.
     - sci_setup: dict with keys "boards" and "sci_channels" for scintillator channels
     - cer_setup: dict with keys "boards" and "cer_channels" for Cherenkov channels
     Returns: sci_setup, cer_setup
     Note: sci_channels and cer_channels should be lists of channel numbers (as strings) 
     corresponding to the channels used for energy sum in the analysis. 
     Boards should be a list of board numbers (as strings) that contain those channels.
     Example JSON structure:
{
    "Boardn": [Channeln]
    '''
    with open("data/channels_sci.json", "r") as f:
        sci_setup = json.load(f)
    with open("data/channels_cer.json", "r") as f:
        cer_setup = json.load(f)
    return sci_setup, cer_setup

def reconstruct_energy(b, ch, HG_matrix, LG_matrix, calib_data, saturation_thresh=7500):
    '''
    Return Calibrated energy using the formula(s):
    if event is not saturated:
        e_cal = (hg - hg pedestal)*(correction factor)
    if event is saturated:
        e_cal = (1/m * (lg - lg pedestal) + b)*(correction factor)
    '''
    c = calib_data[b][str(ch)]
    
    # Select columns
    raw_HG = HG_matrix[:, ch]
    raw_LG = LG_matrix[:, ch]

    # 1. Subtract HG pedestal
    # 2. Apply response factor
    e_hg = (raw_HG - c["ped_HG"]) * c["factor"]

    # 1. Subtract LG pedestal
    # 2. Convert to HG equivalent using m and b
    # 3. Apply final response factor
    lg_corrected = ((raw_LG - c["ped_LG"]) - c["b"]) / c["m"]
    e_lg = lg_corrected * c["factor"]
    return np.where(raw_HG < saturation_thresh, e_hg, e_lg)

def load_energy_data(run, calib_data=False):
    check = check_file(run)
    file_path = get_run_filepath(run)
    sci_setup, cer_setup = load_configs() # Load configs for energy boards and channels once and pass to analysis

    if check:
        with uproot.open(file_path) as f:
            t = f["EventTree"]
            num_events = t.num_entries
            total_sci_energy = np.zeros(num_events)
            total_cer_energy = np.zeros(num_events)

            boards = list(sci_setup.keys())
            if calib_data:
                with open("data/energy_calibration.json", "r") as f:
                    calib_data = json.load(f)
            for b in boards:
                hg_branch = f"FERS_{b}_energyHG"
                lg_branch = f"FERS_{b}_energyLG"

                if hg_branch in t.keys() and lg_branch in t.keys():
                    HG_matrix = np.stack(t[hg_branch].array(library="np"))
                    LG_matrix = np.stack(t[lg_branch].array(library="np"))
                    # Scintillators
                    if calib_data:
                        for ch in [int(c) for c in sci_setup.get(b, [])]:
                            total_sci_energy += reconstruct_energy(b, ch, HG_matrix, LG_matrix, calib_data)

                        # Cherenkovs
                        for ch in [int(c) for c in cer_setup.get(b, [])]:
                            total_cer_energy += reconstruct_energy(b, ch, HG_matrix, LG_matrix, calib_data)
                    else:
                        total_sci_energy += np.sum(HG_matrix[:, [int(c) for c in sci_setup.get(b, [])]], axis=1)

                        total_cer_energy += np.sum(HG_matrix[:, [int(c) for c in cer_setup.get(b, [])]], axis=1)

    else:
        raise ValueError(f"File is not a fers run file: {run}")

    return total_sci_energy, total_cer_energy