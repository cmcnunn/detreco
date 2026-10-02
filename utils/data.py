"""Lookup helpers for the run list (data/run_list.json).

Maps run IDs to the converted ROOT file recorded for them, so scripts don't
each hardcode the path to run_list.json or repeat the same JSON-loading
boilerplate.
"""

import json
from pathlib import Path

RUN_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "run_list.json"


def load_run_list(path=RUN_LIST_PATH):
    """Return the run list as a dict of run ID (str) -> info dict."""
    with open(path) as f:
        return json.load(f)


def get_run_filepath(run_id, path=RUN_LIST_PATH):
    """Return the ROOT file path (str, or list of str for multi-file runs)
    recorded for ``run_id``.

    ``run_id`` may be an int or str. Raises ``KeyError`` if the run isn't in
    the run list.
    """
    run_list = load_run_list(path)
    run_id = str(run_id)
    if run_id not in run_list:
        raise KeyError(f"Run {run_id} not found in {path}")
    return run_list[run_id]["file"]


def get_run_beam(run_id, path=RUN_LIST_PATH):
    """Return the ``(beam_type, beam_energy_gev)`` recorded for ``run_id``.

    Either element may be ``None`` if the run isn't covered by the elog CSV
    that ``scripts/update_run_list.py`` merges in (e.g. TB2025 runs).
    """
    run_list = load_run_list(path)
    run_id = str(run_id)
    if run_id not in run_list:
        raise KeyError(f"Run {run_id} not found in {path}")
    entry = run_list[run_id]
    return entry.get("beam_type"), entry.get("beam_energy_gev")


def get_run_table_position(run_id, path=RUN_LIST_PATH):
    """Return the ``(table_x_mm, table_y_mm)`` recorded for ``run_id``.

    Either element may be ``None`` if the run isn't covered by the elog CSV
    that ``scripts/update_run_list.py`` merges in, or the cell was blank.
    """
    run_list = load_run_list(path)
    run_id = str(run_id)
    if run_id not in run_list:
        raise KeyError(f"Run {run_id} not found in {path}")
    entry = run_list[run_id]
    return entry.get("table_x_mm"), entry.get("table_y_mm")


def get_run_hodo_rotation(run_id, path=RUN_LIST_PATH):
    """Return the hodoscope-vs-tracker cross-slopes ``(dx_dy, dy_dx)`` (rad)
    recorded for ``run_id``, or ``None`` if the run has none (e.g. TB2025
    runs, which have no tracker to measure it against).

    Stored under the run's ``hodo_rotation`` key by ``scripts.hodores
    --update-run-list`` -- ``dx_dy_mrad``/``dy_dx_mrad`` plus ``from_runs``,
    the run(s) whose fit supplied them (the run itself, or the median of the
    nearest runs with a trusted fit). Applied by
    ``utils.hodo.reconstruct_hodoscope(run_id=...)``.
    """
    run_list = load_run_list(path)
    run_id = str(run_id)
    if run_id not in run_list:
        raise KeyError(f"Run {run_id} not found in {path}")
    rotation = run_list[run_id].get("hodo_rotation")
    if rotation is None:
        return None
    return 1e-3 * rotation["dx_dy_mrad"], 1e-3 * rotation["dy_dx_mrad"]


def get_runs_by_beam(beam_type=None, beam_energy_gev=None, path=RUN_LIST_PATH, include_flagged=False):
    """Return the sorted run IDs (as ints) matching the given beam filters.

    Pass ``beam_type`` and/or ``beam_energy_gev`` to narrow the selection;
    omit either to leave that dimension unfiltered. Runs with unknown beam
    metadata (``None``) never match an explicit filter.

    Excludes runs a shifter's elog notes flagged as junk, a documented
    beam-quality problem, or a calibration run (see
    ``scripts/update_run_list.py``'s ``_BAD_RUN_RE``) unless
    ``include_flagged=True`` -- these still carry a normal beam_type/energy
    tag (e.g. "JUNK RUN No current on board 8" tagged e+ 40 GeV, or "PSB
    ring down ... muon contamination" tagged e+ 120 GeV), so without this
    they'd silently get pooled in with clean physics runs at that energy.
    """
    run_list = load_run_list(path)
    matches = []
    for run_id, entry in run_list.items():
        if beam_type is not None and entry.get("beam_type") != beam_type:
            continue
        if beam_energy_gev is not None and entry.get("beam_energy_gev") != beam_energy_gev:
            continue
        if not include_flagged and entry.get("flagged"):
            continue
        matches.append(int(run_id))
    return sorted(matches)


def list_beam_types(path=RUN_LIST_PATH):
    """Return the sorted, distinct non-null beam types present in the run list."""
    run_list = load_run_list(path)
    return sorted({e["beam_type"] for e in run_list.values() if e.get("beam_type")})


def list_beam_energies(beam_type=None, path=RUN_LIST_PATH):
    """Return the sorted, distinct non-null beam energies (GeV), optionally filtered to one beam type."""
    run_list = load_run_list(path)
    energies = {
        e["beam_energy_gev"] for e in run_list.values()
        if e.get("beam_energy_gev") is not None
        and (beam_type is None or e.get("beam_type") == beam_type)
    }
    return sorted(energies)
