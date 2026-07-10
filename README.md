# detreco

This repo is for determining the efficiency and spatial resolution of the tracking system for HG-Dream.

## Setup

on HPCC

```bash
/lustre/work/colnunn/dereco
```

A Python virtual environment is checked into `detreco/` with the required packages
(`uproot`, `awkward`, `numpy`, `scipy`, `matplotlib`, `mplhep`) already installed.

```bash
source detreco/bin/activate
```

Run IDs are resolved to converted ROOT file paths via `data/run_list.json`.

## Usage

Scripts are run from the repo root so they can import the `utils` package:

source ~/.bashrc 

```bash
python scripts/hodoreco.py --run <run_id>
python scripts/sitrackreco.py --run <run_id>
python scripts/effplots.py [--run <run_id>]   # defaults to all runs in run_list.json
python scripts/plot_waveforms.py <root_file> <branch> [-n NEVENTS] [--overlay]
```

Plots are written under `output/` (e.g. `output/sitrackreco/`, `output/effplots/`).

## Layout

- `scripts/` — entry-point analysis scripts, one per task:
  - `hodoreco.py` — reconstructs hit position from the hodoscope (HG fiber planes).
  - `sitrackreco.py` — reconstructs hit position from the silicon strip tracker.
  - `effplots.py` — builds hodoscope-referenced 2D efficiency maps for MCP1, MCP2, and the Veto.
  - `plot_waveforms.py` — plots baseline-subtracted waveforms from a ROOT file for debugging.
- `utils/` — shared helpers used across scripts:
  - `data.py` — run-list lookup (run ID -> ROOT file path).
  - `io.py` — file I/O and multi-run orchestration (multiprocessing over runs).
  - `hodo.py` — hodoscope hit reconstruction logic.
  - `tracker.py` — loader for raw silicon-tracker ASCII dumps.
  - `waveforms.py` — waveform baseline subtraction and related processing.
  - `selectors.py` — boolean event-selection masks (veto, pulse windows, etc.).
  - `plotting.py` — reusable plotting/histogram helpers (e.g. TProfile-style binning).
  - `constants.py` — shared detector constants and thresholds.
- `data/run_list.json` — maps run ID -> path to the converted ROOT file for that run.
- `output/` — generated plots, one subdirectory per script.
