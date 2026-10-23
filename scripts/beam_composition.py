"""Beam makeup per beam type and energy, from the service-DRS PID counters.

Classifies every event of every run with ``utils.pid`` (muon / electron /
pion / proton, the CaloXDataAnalysis selection; anything else is
"unclassified") and pools the counts by beam type and energy, both over all
triggers and over hole-veto-passing events -- the reference deteff_scan.py
measures efficiencies on. The nominal beam label isn't the particle mix: the
e+ 120-160 GeV beams are mostly muons, for example.

See ``utils.pid`` for how far each class can be trusted: the muon tag is
clean, while the electron/pion split depends on the Cherenkov pressures
(e.g. 12-17% of e+ 10-80 GeV events land in "pion" with full electron
energy).

Writes per-run, per-energy and per-beam-type CSVs, a fraction-vs-beam-type
plot (pooled over energy, so dominated by whichever energies have the most
events), and one fraction-vs-energy plot per beam type -- all veto-passing.

Usage:
    python3 -m scripts.beam_composition [--beam-type e+ ...] [--include-flagged] [--workers N]
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot

from utils.data import get_run_beam, get_run_filepath, get_runs_by_beam, list_beam_energies, list_beam_types
from utils.io import ensure_output_dir
from utils.pid import DEFAULT_STEP_SIZE, PARTICLE_REQUIREMENTS, load_particle_masks
from utils.plotting import BEAM_LABELS, BEAM_STYLE, OTHER_BEAM_STYLE, ordered_beam_types, style_summary_axes
from utils.selectors import get_branch_names, passes_veto
from utils.si_efficiency import MIN_EVENTS_FOR_SCAN, run_event_count

OUTPUT_DIR = ensure_output_dir("beam_composition")

# One worker per CPU this process may use (respects a SLURM/cgroup
# allocation); each holds one 10k-event chunk of four DRS channels (~160 MB).
N_WORKERS = len(os.sched_getaffinity(0))

CLASSES = tuple(PARTICLE_REQUIREMENTS) + ("unclassified",)
REFERENCES = ("veto", "all")  # veto-passing events, every trigger

# Each particle drawn in the style of the beam it's named for.
PARTICLE_STYLE = {
    "electron": BEAM_STYLE["e+"],
    "pion": BEAM_STYLE["pi+"],
    "muon": BEAM_STYLE["mu+"],
    "proton": OTHER_BEAM_STYLE,
}
PARTICLE_LABELS = {"electron": "Electron", "pion": "Pion", "muon": "Muon", "proton": "Proton"}
# Sideways nudges so the four particles at one energy/beam type don't overlap.
ENERGY_NUDGE_GEV = {"electron": -3.0, "pion": -1.0, "muon": 1.0, "proton": 3.0}
CATEGORY_NUDGE = {"electron": -0.21, "pion": -0.07, "muon": 0.07, "proton": 0.21}


def _load_veto_mask(run_id, step_size=DEFAULT_STEP_SIZE):
    """Hole-veto pass mask over every ROOT event, read in chunks like utils.pid."""
    veto_branch, _, _ = get_branch_names(run_id)
    filepaths = get_run_filepath(run_id)
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    return np.concatenate([
        passes_veto(np.stack(arrs[veto_branch]))
        for arrs in uproot.iterate([f"{fp}:EventTree" for fp in filepaths], [veto_branch],
                                   step_size=step_size, library="np")
    ])


def run_counts(run_id):
    """``{reference: {"n": events, class: count}}`` for one run (see REFERENCES/CLASSES)."""
    masks = load_particle_masks(run_id)
    masks["unclassified"] = ~np.logical_or.reduce(list(masks.values()))
    selections = {"veto": _load_veto_mask(run_id)}
    selections["all"] = np.ones_like(selections["veto"])
    return {ref: {"n": int(sel.sum()), **{c: int((masks[c] & sel).sum()) for c in CLASSES}}
            for ref, sel in selections.items()}


def _load_one(run_id):
    """Picklable per-run worker for the process pool."""
    try:
        return run_id, run_counts(str(run_id)), None
    except Exception as e:
        return run_id, None, str(e)


def load_all_runs(run_ids, workers):
    """``(results, skipped)``: ``{run_id: run_counts}`` and ``{run_id: reason}``."""
    results, skipped = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_load_one, run_id) for run_id in run_ids]
        for i, fut in enumerate(as_completed(futures), 1):
            run_id, counts, err = fut.result()
            if err is None:
                results[run_id] = counts
            else:
                skipped[run_id] = err
                print(f"    [skip] run {run_id}: {err}")
            print(f"  {i}/{len(run_ids)} runs loaded", end="\r", flush=True)
    print()
    return results, skipped


def pool(counts_list):
    """Sum a list of run_counts dicts."""
    return {ref: {key: sum(c[ref][key] for c in counts_list) for key in ("n",) + CLASSES}
            for ref in REFERENCES}


def fraction_columns(counts):
    """``n_<ref>`` and ``frac_<class>_<ref>`` columns from (pooled) counts."""
    cols = {}
    for ref in REFERENCES:
        n = counts[ref]["n"]
        cols[f"n_{ref}"] = n
        for c in CLASSES:
            cols[f"frac_{c}_{ref}"] = counts[ref][c] / n if n else 0.0
    return cols


def _format_fractions(cols, ref="veto"):
    return "  ".join(f"{c}={100 * cols[f'frac_{c}_{ref}']:5.1f}%" for c in CLASSES)


def write_csv(rows, filename):
    if not rows:
        print(f"No rows to write to {filename}.")
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {filename}")


def _plot_fractions(ax, x, rows, particle):
    """One particle's veto-passing fraction, with a binomial error bar."""
    frac = np.array([r[f"frac_{particle}_veto"] for r in rows])
    n = np.array([r["n_veto"] for r in rows], dtype=float)
    err = np.sqrt(frac * (1 - frac) / np.maximum(n, 1))
    ax.errorbar(x, frac, yerr=err, ls="none", ms=9, elinewidth=1.5, capsize=0,
                label=PARTICLE_LABELS[particle], **PARTICLE_STYLE[particle])


def _finish_axes(ax, rlabel, filename):
    ax.set_ylabel("Fraction of events", loc="top")
    ax.set_ylim(0, 1.05)
    style_summary_axes(ax)
    ax.legend(loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    mh.label.exp_label(exp="CaloX", text="Preliminary", data=True, rlabel=rlabel, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved {filename}")
    plt.close()


def plot_vs_beamtype(rows, filename):
    """Veto-passing particle fractions per nominal beam type, pooled over energy."""
    if not rows:
        return
    by_type = {r["beam_type"]: r for r in rows}
    beam_types = ordered_beam_types(by_type)
    ordered = [by_type[b] for b in beam_types]
    x = np.arange(len(beam_types))
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 8))
    for particle in PARTICLE_REQUIREMENTS:
        _plot_fractions(ax, x + CATEGORY_NUDGE[particle], ordered, particle)
    ax.set_xticks(x)
    ax.set_xticklabels([BEAM_LABELS.get(b, b) for b in beam_types], fontsize=26)
    ax.set_xlim(-0.6, len(beam_types) - 0.4)
    ax.set_xlabel("Nominal beam", loc="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    _finish_axes(ax, "Beam composition", filename)


def plot_vs_energy(rows, beam_type, filename):
    """Veto-passing particle fractions vs energy for one nominal beam type."""
    sub = sorted((r for r in rows if r["beam_type"] == beam_type), key=lambda r: r["beam_energy_gev"])
    if not sub:
        return
    energies = np.array([r["beam_energy_gev"] for r in sub], dtype=float)
    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 8))
    for particle in PARTICLE_REQUIREMENTS:
        _plot_fractions(ax, energies + ENERGY_NUDGE_GEV[particle], sub, particle)
    ax.set_xlim(0, 180)
    ax.set_xlabel("Beam energy [GeV]", loc="right")
    _finish_axes(ax, f"Nominal {BEAM_LABELS.get(beam_type, beam_type)} beam", filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beam-type", action="append", default=None,
                        help="Only this nominal beam type (repeatable); default: every one in run_list.json.")
    parser.add_argument("--include-flagged", action="store_true",
                        help="Also include runs the elog flagged (junk, beam problems, calibration).")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel run loaders (default: {N_WORKERS}, one per available CPU).")
    args = parser.parse_args()

    beam_types = args.beam_type or [b for b in list_beam_types() if b != "pedestal"]
    plan = [(beam_type, energy, get_runs_by_beam(beam_type, energy, include_flagged=args.include_flagged))
            for beam_type in beam_types for energy in list_beam_energies(beam_type)]
    all_run_ids = sorted({run_id for _, _, run_ids in plan for run_id in run_ids})

    # Short calibration/threshold-scan blips carry a physics beam tag too; drop
    # them up front, as deteff_scan.py does.
    skipped = {}
    run_ids = []
    for run_id in all_run_ids:
        n = run_event_count(str(run_id))
        if n < MIN_EVENTS_FOR_SCAN:
            skipped[run_id] = f"only {n} events (< {MIN_EVENTS_FOR_SCAN})"
        else:
            run_ids.append(run_id)
    print(f"Loading {len(run_ids)} run(s) across {len(plan)} (beam type, energy) groups "
          f"using {args.workers} worker(s)...")
    results, skipped_load = load_all_runs(run_ids, args.workers)
    skipped.update(skipped_load)
    if skipped:
        path = os.path.join(OUTPUT_DIR, "skipped_runs.log")
        with open(path, "w") as f:
            for run_id, reason in sorted(skipped.items()):
                f.write(f"{run_id}: {reason}\n")
        print(f"{len(skipped)}/{len(all_run_ids)} run(s) skipped; reasons in {path}")

    run_rows = []
    for run_id in sorted(results):
        beam_type, energy = get_run_beam(run_id)
        run_rows.append({"run": run_id, "beam_type": beam_type, "beam_energy_gev": energy,
                         **fraction_columns(results[run_id])})

    energy_rows, beamtype_rows = [], []
    for beam_type in ordered_beam_types(beam_types):
        group = [(e, [r for r in ids if r in results]) for bt, e, ids in plan if bt == beam_type]
        group = [(e, ids) for e, ids in group if ids]
        if not group:
            continue
        print(f"=== nominal {beam_type} (veto-passing) ===")
        for energy, ids in group:
            cols = fraction_columns(pool([results[r] for r in ids]))
            energy_rows.append({"beam_type": beam_type, "beam_energy_gev": energy, "n_runs": len(ids), **cols})
            print(f"  {energy:>4} GeV  runs={len(ids):3d}  n={cols['n_veto']:8d}  {_format_fractions(cols)}")
        cols = fraction_columns(pool([results[r] for _, ids in group for r in ids]))
        beamtype_rows.append({"beam_type": beam_type, "n_runs": sum(len(ids) for _, ids in group), **cols})
        print(f"  all energies  n={cols['n_veto']:8d}  {_format_fractions(cols)}")

    write_csv(run_rows, os.path.join(OUTPUT_DIR, "composition_by_run.csv"))
    write_csv(energy_rows, os.path.join(OUTPUT_DIR, "composition_by_energy.csv"))
    write_csv(beamtype_rows, os.path.join(OUTPUT_DIR, "composition_by_beamtype.csv"))
    plot_vs_beamtype(beamtype_rows, os.path.join(OUTPUT_DIR, "composition_vs_beamtype.png"))
    for beam_type in ordered_beam_types(beam_types):
        plot_vs_energy(energy_rows, beam_type, os.path.join(OUTPUT_DIR, f"composition_vs_energy_{beam_type}.png"))


if __name__ == "__main__":
    main()
