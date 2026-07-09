"""File-I/O and multi-run orchestration helpers.

The bulk of the scripts share the same shape:
 1. load ``data/run_list.json`` which maps run-id -> {"file": "/path/to/root"}
 2. validate each run's ROOT file exists and has entries
 3. fan out the per-run work with a ``Pool`` and aggregate the results.

This module captures that pattern once so scripts can focus on the per-run
analysis logic.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

# ``tqdm`` is optional; fall back to a no-op if it isn't installed.
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it


DEFAULT_RUN_LIST_PATH = "data/run_list.json"


# ---------------------------------------------------------------------------
# Project root + paths
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Directory that contains ``utils/`` (= the project root).

    Robust to being imported from any cwd: scripts can now resolve input
    files by path relative to the project rather than the current directory.
    """
    return Path(__file__).resolve().parent.parent


def ensure_output_dir(subdir: str | os.PathLike = "") -> str:
    """Create (if missing) and return the absolute path of ``output/<subdir>/``."""
    base = project_root() / "output"
    path = base / subdir if subdir else base
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# Run-list
# ---------------------------------------------------------------------------
def load_run_list(path: str | os.PathLike = DEFAULT_RUN_LIST_PATH) -> dict:
    """Load ``run_list.json`` (keys are run-ids, values at minimum have 'file')."""
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with open(path, "r") as f:
        return json.load(f)


def get_run_file(run_id: str, run_list: dict) -> Optional[str]:
    """Look up a run's ROOT-file path in ``run_list``. Returns ``None`` if absent."""
    entry = run_list.get(str(run_id))
    if not entry:
        return None
    return entry.get("file")


def check_root_file(file_path: str | os.PathLike, run_id: str | None = None,
                    verbose: bool = True) -> bool:
    """Return True if the file exists on disk (optionally log a warning)."""
    exists = Path(file_path).exists()
    if not exists and verbose:
        tag = f"run {run_id}" if run_id is not None else "file"
        print(f"[WARN] Missing {tag}: {file_path}")
    return exists


def resolve_run_files(run_ids: Iterable[str],
                      run_list: dict | None = None) -> list[tuple[str, str]]:
    """Return the ``(run_id, file_path)`` pairs for runs that exist on disk.

    Convenient guard to put at the top of a ``main()`` so you don't scatter
    existence checks throughout the script.
    """
    if run_list is None:
        run_list = load_run_list()
    out = []
    for run_id in run_ids:
        path = get_run_file(run_id, run_list)
        if path and check_root_file(path, run_id=run_id):
            out.append((str(run_id), path))
    return out


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
def parallel_process_runs(run_ids: Iterable[str],
                          process_func: Callable,
                          run_list: dict | None = None,
                          max_workers: int | None = None,
                          desc: str = "Processing runs",
                          skip_none: bool = True) -> list:
    """Run ``process_func(run_id, file_path)`` across ``run_ids`` in parallel.

    Validates each file exists first, then fans out with a
    ``ProcessPoolExecutor``. Returns the list of results in completion order;
    ``None`` returns are dropped when ``skip_none`` is True.

    ``process_func`` must be picklable (defined at module scope).
    """
    pairs = resolve_run_files(run_ids, run_list=run_list)
    if not pairs:
        return []

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_func, rid, path): rid for rid, path in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            res = fut.result()
            if skip_none and res is None:
                continue
            results.append(res)
    return results
