"""Si-tracker efficiency scan across beam type, then beam energy.

For each beam type (e+, pi+, mu+, ...) and each energy within it, pools
every matching run's hodoscope-referenced hits and computes the intrinsic
efficiency of each si-tracker station, plus the hodoscope's own event-level
reconstruction rate (n_good_hodo / n_events, not gated by the reference
selection). The hodoscope numbers are pooled over every run with a readable
ROOT file, independent of whether that run's tracker data is available or
aligns cleanly (see utils.si_efficiency.load_si_and_hodo) -- si1/si2 still
necessarily need a working tracker, so they're pooled over the smaller
subset that provides. Produces two summary plots -- efficiency vs. beam type
(pooled over energy) and efficiency vs. beam energy (one line per beam
type) -- plus the underlying numbers as CSVs.

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

from scipy.stats import beta

from utils.data import get_runs_by_beam, list_beam_energies, list_beam_types
from utils.io import ensure_output_dir
from utils.si_efficiency import MIN_EVENTS_FOR_SCAN, load_si_and_hodo, run_event_count

OUTPUT_DIR = ensure_output_dir("deteff_scan")

MARKERS = {"e+": "o", "pi+": "s", "mu+": "^"}
STATION_STYLE = {
    "si1": dict(ls="-", fillstyle="full"),
    "si2": dict(ls="--", fillstyle="none"),
    "hodo": dict(ls=":", fillstyle="full"),
}
STATION_COLORS = {"si1": "blue", "si2": "green", "hodo": "red"}
STATION_LABELS = {"si1": "Si Tracker 1", "si2": "Si Tracker 2", "hodo": "Hodoscope"}

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


def filter_scannable_runs(run_ids):
    """Drop runs too short to possibly align before the expensive load step.

    A handful of "runs" in run_list.json are short calibration/threshold-scan
    blips (tens of events) tagged with a physics beam_type/energy anyway --
    too short to contain the >= 5 spill boundaries
    align_tracker_to_root_by_timestamp needs to fit a clock relationship, so
    they'd fail the load regardless. Filtering by event count (cheap:
    metadata-only) up front avoids paying for the full tracker load +
    alignment just to hit that same failure, and -- more importantly --
    surfaces them as an explicit, logged skip rather than leaving "which
    runs happened to be long enough" as a silent, energy-correlated bias in
    which runs make it into the pooled average.

    Returns ``(keep, skipped)``: ``keep`` is the run ids to actually load;
    ``skipped`` is ``{run_id: reason}`` for the rest.
    """
    keep, skipped = [], {}
    for run_id in run_ids:
        try:
            n = run_event_count(str(run_id))
        except Exception as e:
            skipped[run_id] = f"couldn't read event count: {e}"
            continue
        if n < MIN_EVENTS_FOR_SCAN:
            skipped[run_id] = f"only {n} events (< {MIN_EVENTS_FOR_SCAN}), too short to align"
        else:
            keep.append(run_id)
    return keep, skipped


def load_all_runs(run_ids):
    """Load every run in ``run_ids`` in parallel.

    Returns ``(results, skipped)``: ``results`` is ``{run_id: result}`` for
    runs that loaded successfully; ``skipped`` is ``{run_id: reason}`` for
    the rest (bad alignment, missing tracker files, etc.).
    """
    results = {}
    skipped = {}
    with tqdm(total=len(run_ids), desc="Scanning runs", unit="run") as pbar, \
        ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_one, run_id): run_id for run_id in run_ids}
        for fut in as_completed(futures):
            run_id, result, err = fut.result()
            pbar.set_postfix_str(f"run {run_id}")
            if err is not None:
                skipped[run_id] = err
                tqdm.write(f"    [skip] run {run_id}: {err}")
            else:
                results[run_id] = result
            pbar.update(1)
    return results, skipped


def pool_group(run_ids, results_by_run):
    """Pool already-loaded results for ``run_ids``.

    Returns ``(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good,
    n_events, n_hodo_runs, n_si_runs)`` or ``None`` if none of the runs in
    the group loaded successfully.

    The hodoscope counts are summed over every loaded run regardless of
    whether its tracker aligned (``load_si_and_hodo`` returns valid
    n_hodo_good/n_events straight from ROOT even when tracker_error is set)
    -- ``n_hodo_runs`` is how many runs that covers. The six position arrays
    only get a real contribution from runs whose tracker did align (a
    tracker_error run contributes empty arrays), so si1/si2 efficiency is
    unaffected by tracker-unavailable runs; ``n_si_runs`` is that smaller
    count.
    """
    parts = [results_by_run[r] for r in run_ids if r in results_by_run]
    if not parts:
        return None
    arrays = tuple(np.concatenate(arrs) for arrs in zip(*(p[:6] for p in parts)))
    n_hodo_good = sum(p[6] for p in parts)
    n_events = sum(p[7] for p in parts)
    n_si_runs = sum(1 for p in parts if p[8] is None)
    return arrays + (n_hodo_good, n_events, len(parts), n_si_runs)


# 1-sigma equivalent, matching the previous normal-approximation's implicit
# confidence level (sqrt(p(1-p)/n) is a 1-sigma standard error).
_CL = 0.6827


def _binomial_efficiency(n_pass, n_total):
    """Plain event-count ratio (numerator = ref events with a hit on the
    station, denominator = all ref events), same as sieff.py's main() -- no
    binning or geometric-footprint masking.

    Uncertainty is the Clopper-Pearson exact interval rather than the normal
    (Wald) approximation: the latter collapses to +/-0 at eff == 0 or 1,
    which is misleading for the near-100% efficiencies these trackers
    typically show. Collapsed to a single symmetric-ish value (the larger of
    the two half-widths) so callers can keep treating this as eff +/- unc.
    """
    if n_total == 0:
        return 0.0, None
    eff = n_pass / n_total
    alpha = 1 - _CL
    lower = beta.ppf(alpha / 2, n_pass, n_total - n_pass + 1) if n_pass > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, n_pass + 1, n_total - n_pass) if n_pass < n_total else 1.0
    unc = max(eff - lower, upper - eff)
    return eff, unc


def station_efficiencies(xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2):
    n_ref = len(xh_ref)
    eff1_mean, eff1_unc = _binomial_efficiency(len(xh_sel1), n_ref)
    eff2_mean, eff2_unc = _binomial_efficiency(len(xh_sel2), n_ref)
    return eff1_mean, eff1_unc, eff2_mean, eff2_unc


def hodo_efficiency(n_hodo_good, n_events):
    return _binomial_efficiency(n_hodo_good, n_events)


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
    scannable_run_ids, skipped_short = filter_scannable_runs(all_run_ids)
    if skipped_short:
        print(f"Skipping {len(skipped_short)}/{len(all_run_ids)} run(s) below "
              f"{MIN_EVENTS_FOR_SCAN} events (too short to align) before loading.")
    print(f"Loading {len(scannable_run_ids)} run(s) across {len(plan)} (beam type, energy) groups "
          f"using {N_WORKERS} worker process(es)...")
    results_by_run, skipped_load = load_all_runs(scannable_run_ids)

    skipped_all = {**skipped_short, **skipped_load}
    if skipped_all:
        skip_log_path = os.path.join(OUTPUT_DIR, "skipped_runs.log")
        with open(skip_log_path, "w") as f:
            for run_id, reason in sorted(skipped_all.items()):
                f.write(f"{run_id}: {reason}\n")
        print(f"{len(skipped_all)}/{len(all_run_ids)} run(s) skipped total; reasons written to {skip_log_path}")

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
            xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2, n_hodo_good, n_events, n_hodo_runs, n_si_runs = result
            pooled_parts.append((xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2,
                                n_hodo_good, n_events))

            eff1_mean, eff1_unc, eff2_mean, eff2_unc = station_efficiencies(
                xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2)
            hodo_eff, hodo_unc = hodo_efficiency(n_hodo_good, n_events)
            print(f"  {energy} GeV: n_hodo_runs={n_hodo_runs}/{len(run_ids)}  n_si_runs={n_si_runs}  "
                  f"n_ref={len(xh_ref)}  "
                  f"si1={eff1_mean:.3f}±{(eff1_unc or 0):.3f}  "
                  f"si2={eff2_mean:.3f}±{(eff2_unc or 0):.3f}  "
                  f"hodo={hodo_eff:.3f}±{(hodo_unc or 0):.3f}")

            energy_rows.append({
                "beam_type": beam_type, "beam_energy_gev": energy,
                "n_candidate_runs": len(run_ids), "n_hodo_runs": n_hodo_runs, "n_si_runs": n_si_runs,
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
