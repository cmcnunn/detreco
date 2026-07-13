"""Si-tracker efficiency scan across beam type, then beam energy.

For each beam type (e+, pi+, mu+, ...) and each energy within it, pools
every matching run's hodoscope-referenced hits and computes the intrinsic
efficiency of each si-tracker station, plus the hodoscope's own event-level
reconstruction rate (n_good_hodo / n_events, not gated by the reference
selection). Produces two summary plots -- efficiency vs. beam type (pooled
over energy) and efficiency vs. beam energy (one line per beam type) --
plus the underlying numbers as CSVs.

Runs are loaded in parallel (one process per run, sized to the node's
available CPUs) since each run's ROOT + tracker load/alignment is
independent and this is normally the bottleneck.

Usage:
    python scripts/deteff_scan.py
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
from matplotlib.lines import Line2D

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:
        """No-op stand-in so the scan still runs without the optional dependency."""
        def __init__(self, *a, total=None, **k):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        write = staticmethod(print)

from scripts.sieff import load_si_and_hodo
from utils.data import get_runs_by_beam, list_beam_energies, list_beam_types
from utils.io import ensure_output_dir
from utils.plotting import compute_efficiency_map, intrinsic_efficiency

OUTPUT_DIR = ensure_output_dir("deteff_scan")

MARKERS = {"e+": "o", "pi+": "s", "mu+": "^"}
STATION_STYLE = {
    "si1": dict(ls="-", fillstyle="full"),
    "si2": dict(ls="--", fillstyle="none"),
    "hodo": dict(ls=":", fillstyle="full"),
}
STATION_COLORS = {"si1": "blue", "si2": "green", "hodo": "red"}
STATION_LABELS = {"si1": "Si Tracker 1", "si2": "Si Tracker 2", "hodo": "Hodoscope"}

# Bins with eff <= this are treated as outside the tracker's geometric
# footprint and excluded from the intrinsic-efficiency average (see
# utils.plotting.intrinsic_efficiency). Since this compares against the
# same eff it's conditioning on, a real inefficient-but-in-footprint region
# below this value gets misread as "outside" and dropped -- lowering it
# trades that bias for more edge/noise bins leaking into the average.
MIN_EFF_FOR_FOOTPRINT = 0.1

# One worker per available CPU (respects a SLURM/cgroup allocation, unlike
# os.cpu_count()); bump your job's cpu allocation and this picks it up
# automatically, no flag needed.
N_WORKERS = len(os.sched_getaffinity(0))


def _load_one(run_id):
    """Picklable per-run worker for the process pool."""
    try:
        return run_id, load_si_and_hodo(str(run_id)), None
    except Exception as e:
        return run_id, None, str(e)


def load_all_runs(run_ids):
    """Load every run in ``run_ids`` in parallel. Returns {run_id: result}, skipping failures."""
    results = {}
    with tqdm(total=len(run_ids), desc="Scanning runs", unit="run") as pbar, \
        ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_one, run_id): run_id for run_id in run_ids}
        for fut in as_completed(futures):
            run_id, result, err = fut.result()
            pbar.set_postfix_str(f"run {run_id}")
            if err is not None:
                tqdm.write(f"    [skip] run {run_id}: {err}")
            else:
                results[run_id] = result
            pbar.update(1)
    return results


def pool_group(run_ids, results_by_run):
    """Pool already-loaded results for ``run_ids``.

    Returns ``(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good,
    n_events, n_ok)`` or ``None`` if none of the runs in the group loaded
    successfully. The six position arrays are concatenated; the hodoscope
    counts are summed (they're per-run scalars, not arrays).
    """
    parts = [results_by_run[r] for r in run_ids if r in results_by_run]
    if not parts:
        return None
    arrays = tuple(np.concatenate(arrs) for arrs in zip(*(p[:6] for p in parts)))
    n_hodo_good = sum(p[6] for p in parts)
    n_events = sum(p[7] for p in parts)
    return arrays + (n_hodo_good, n_events, len(parts))


def station_efficiencies(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2):
    eff1, h_ref1, *_ = compute_efficiency_map(xh_ref, yh_ref, xh_sel1, yh_sel1)
    eff2, h_ref2, *_ = compute_efficiency_map(xh_ref, yh_ref, xh_sel2, yh_sel2)
    eff1_mean, eff1_unc = intrinsic_efficiency(eff1, h_ref1, min_eff=MIN_EFF_FOR_FOOTPRINT)
    eff2_mean, eff2_unc = intrinsic_efficiency(eff2, h_ref2, min_eff=MIN_EFF_FOR_FOOTPRINT)
    return eff1_mean, eff1_unc, eff2_mean, eff2_unc


def hodo_efficiency(n_hodo_good, n_events):
    """Plain event-count ratio (+ binomial uncertainty) -- no geometric-footprint
    masking, since this isn't restricted to a reference sample at all."""
    if n_events == 0:
        return 0.0, None
    eff = n_hodo_good / n_events
    unc = np.sqrt(eff * (1 - eff) / n_events)
    return eff, unc


def plot_eff_vs_beamtype(rows, filename):
    if not rows:
        print("No beam-type rows to plot.")
        return
    beam_types = [r["beam_type"] for r in rows]
    x = np.arange(len(beam_types))
    width = 0.25

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x - width, [r["eff_si1"] for r in rows], width,
          yerr=[r["eff_si1_unc"] or 0 for r in rows], capsize=4, label="Si Tracker 1", color=STATION_COLORS["si1"])
    ax.bar(x, [r["eff_si2"] for r in rows], width,
          yerr=[r["eff_si2_unc"] or 0 for r in rows], capsize=4, label="Si Tracker 2", color=STATION_COLORS["si2"])
    ax.bar(x + width, [r["eff_hodo"] for r in rows], width,
          yerr=[r["eff_hodo_unc"] or 0 for r in rows], capsize=4, label="Hodoscope", color=STATION_COLORS["hodo"])
    ax.set_xticks(x)
    ax.set_xticklabels(beam_types)
    ax.set_ylabel("Detector efficiency", loc="top")
    ax.set_ylim(0, 1.05)
    ax.legend()
    mh.label.exp_label(exp="CaloX", data=True, rlabel="Tracker Efficiency by Beam Type", ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Efficiency-vs-beam-type plot saved {filename}")
    plt.close()


def plot_eff_vs_energy(rows, filename):
    """Efficiency vs. energy, broken out by beam type.

    Beam type is encoded as marker shape and station as color/linestyle, so
    a 3-beam-type x 2-station grid needs only two small legends (3 + 2
    entries) instead of one combined 6-entry legend.
    """
    if not rows:
        print("No energy rows to plot.")
        return
    beam_types = sorted({r["beam_type"] for r in rows})

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 8))
    for beam_type in beam_types:
        sub = sorted((r for r in rows if r["beam_type"] == beam_type),
                    key=lambda r: r["beam_energy_gev"])
        energies = [r["beam_energy_gev"] for r in sub]
        for station_key in ("si1", "si2", "hodo"):
            eff = [r[f"eff_{station_key}"] for r in sub]
            unc = [r[f"eff_{station_key}_unc"] or 0 for r in sub]
            ax.errorbar(energies, eff, yerr=unc, capsize=3,
                       marker=MARKERS.get(beam_type, "o"), color=STATION_COLORS[station_key],
                       ls=STATION_STYLE[station_key]["ls"])

    beamtype_handles = [
        Line2D([], [], marker=MARKERS.get(bt, "o"), color="black", ls="none", ms=10, label=bt)
        for bt in beam_types
    ]
    station_handles = [
        Line2D([], [], color=STATION_COLORS[sk], ls=STATION_STYLE[sk]["ls"], label=STATION_LABELS[sk])
        for sk in ("si1", "si2", "hodo")
    ]
    beamtype_legend = ax.legend(handles=beamtype_handles, title="Beam type", loc="lower left")
    ax.add_artist(beamtype_legend)
    ax.legend(handles=station_handles, loc="lower right")

    ax.set_xlabel("Beam energy [GeV]", loc="right")
    ax.set_ylabel("Detector efficiency", loc="top")
    ax.set_ylim(0, 1.05)
    mh.label.exp_label(exp="CaloX", data=True, rlabel="Tracker Efficiency vs Energy", ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Efficiency-vs-energy plot saved {filename}")
    plt.close()


def write_summary(rows, filename):
    if not rows:
        print(f"No rows to write to {filename}.")
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {filename}")


def main():
    beam_types = [b for b in list_beam_types() if b != "pedestal"]

    # Resolve the full (beam_type, energy, run_ids) plan up front, then load
    # every distinct run exactly once, in parallel, before doing any pooling.
    plan = [
        (beam_type, energy, get_runs_by_beam(beam_type, energy))
        for beam_type in beam_types
        for energy in list_beam_energies(beam_type)
    ]
    all_run_ids = sorted({run_id for _, _, run_ids in plan for run_id in run_ids})
    print(f"Loading {len(all_run_ids)} run(s) across {len(plan)} (beam type, energy) groups "
          f"using {N_WORKERS} worker process(es)...")
    results_by_run = load_all_runs(all_run_ids)

    energy_rows = []
    beamtype_rows = []

    for beam_type in beam_types:
        group_plan = [(e, r) for bt, e, r in plan if bt == beam_type]
        if not group_plan:
            print(f"No energies found for beam type {beam_type!r}, skipping.")
            continue
        print(f"=== {beam_type} ({len(group_plan)} energies) ===")

        pooled_parts = []
        for energy, run_ids in group_plan:
            result = pool_group(run_ids, results_by_run)
            if result is None:
                print(f"  {energy} GeV: no usable runs (of {len(run_ids)}), skipping.")
                continue
            xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good, n_events, n_ok = result
            pooled_parts.append((xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2,
                                n_hodo_good, n_events))

            eff1_mean, eff1_unc, eff2_mean, eff2_unc = station_efficiencies(
                xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2)
            hodo_eff, hodo_unc = hodo_efficiency(n_hodo_good, n_events)
            print(f"  {energy} GeV: n_runs={n_ok}/{len(run_ids)}  n_ref={len(xh_ref)}  "
                  f"si1={eff1_mean:.3f}±{(eff1_unc or 0):.3f}  "
                  f"si2={eff2_mean:.3f}±{(eff2_unc or 0):.3f}  "
                  f"hodo={hodo_eff:.3f}±{(hodo_unc or 0):.3f}")

            energy_rows.append({
                "beam_type": beam_type, "beam_energy_gev": energy, "n_runs": n_ok,
                "n_ref_events": len(xh_ref), "n_events": n_events,
                "eff_si1": eff1_mean, "eff_si1_unc": eff1_unc,
                "eff_si2": eff2_mean, "eff_si2_unc": eff2_unc,
                "eff_hodo": hodo_eff, "eff_hodo_unc": hodo_unc,
            })

        if not pooled_parts:
            continue
        xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2 = (
            np.concatenate(arrs) for arrs in zip(*(p[:6] for p in pooled_parts))
        )
        n_hodo_good_total = sum(p[6] for p in pooled_parts)
        n_events_total = sum(p[7] for p in pooled_parts)
        eff1_mean, eff1_unc, eff2_mean, eff2_unc = station_efficiencies(
            xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2)
        hodo_eff, hodo_unc = hodo_efficiency(n_hodo_good_total, n_events_total)
        print(f"  [{beam_type} all energies] n_ref={len(xh_ref)}  "
              f"si1={eff1_mean:.3f}±{(eff1_unc or 0):.3f}  "
              f"si2={eff2_mean:.3f}±{(eff2_unc or 0):.3f}  "
              f"hodo={hodo_eff:.3f}±{(hodo_unc or 0):.3f}")
        beamtype_rows.append({
            "beam_type": beam_type, "n_ref_events": len(xh_ref), "n_events": n_events_total,
            "eff_si1": eff1_mean, "eff_si1_unc": eff1_unc,
            "eff_si2": eff2_mean, "eff_si2_unc": eff2_unc,
            "eff_hodo": hodo_eff, "eff_hodo_unc": hodo_unc,
        })

    plot_eff_vs_beamtype(beamtype_rows, os.path.join(OUTPUT_DIR, "eff_vs_beamtype.png"))
    plot_eff_vs_energy(energy_rows, os.path.join(OUTPUT_DIR, "eff_vs_energy.png"))
    write_summary(energy_rows, os.path.join(OUTPUT_DIR, "summary_by_energy.csv"))
    write_summary(beamtype_rows, os.path.join(OUTPUT_DIR, "summary_by_beamtype.csv"))


if __name__ == "__main__":
    main()
