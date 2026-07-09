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
    """Return the ROOT file path recorded for ``run_id``.

    ``run_id`` may be an int or str. Raises ``KeyError`` if the run isn't in
    the run list.
    """
    run_list = load_run_list(path)
    run_id = str(run_id)
    if run_id not in run_list:
        raise KeyError(f"Run {run_id} not found in {path}")
    return run_list[run_id]["file"]
