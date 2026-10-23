"""Si-tracker and hodoscope efficiency scan across beam type, then beam energy.

For each beam type (e+, pi+, mu+, ...) and each energy within it, pools
every matching run's counts and computes each detector's efficiency with
the same definition: events with a hit on that detector over veto-passing
events. For si1/si2 a hit is a station hit (x and y); for the hodoscope it
is a good hit (one contiguous cluster in each plane). Neither is gated on
the other (see utils.si_efficiency.load_si_and_hodo). The hodoscope numbers
are pooled over every run with a readable ROOT file, independent of whether
that run's tracker data is available or aligns cleanly -- si1/si2
necessarily need a working tracker, so they're pooled over the smaller
subset that provides. Produces two summary plots -- efficiency vs. beam type
(pooled over energy) and efficiency vs. beam energy (one line per beam
type) -- plus the underlying numbers as CSVs.

The plotted hodoscope efficiency is fiducial (hodo_fid): only reference
events whose Si track lands inside its bars count, since the e+/mu+ beams
sat on its Y edge and ~10% of their tracks miss it -- events it couldn't
have reconstructed (see utils.si_efficiency._hodo_fiducial_mask). That
needs a tracker, so it pools over the si runs; the CSVs keep the plain
veto-referenced rate as eff_hodo alongside it.

The nominal beams are mixed (e+ 120-160 GeV is 75-90% muons, pi+ 160 GeV
~23% muons; see scripts/beam_composition.py). With --pid each run's
reference is narrowed to the events the beam-line PID tags as its nominal
beam's own particle -- electrons for e+, pions or protons for pi+, muons for
mu+ (utils.pid.BEAM_PARTICLES) -- and the outputs get a "_pid" suffix, so
the nominal-beam results aren't overwritten. This reads three more DRS
channels per run.

Runs are loaded in parallel (one process per run, sized to the node's
available CPUs) since each run's ROOT + tracker load/alignment is
independent and this is normally the bottleneck.

Usage:
    python3 -m scripts.deteff_scan [--pid]
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np

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

from utils.data import get_run_beam, get_runs_by_beam, list_beam_energies, list_beam_types
from utils.io import ensure_output_dir
from utils.plotting import (BEAM_LABELS, BEAM_STYLE, ENERGY_NUDGE_GEV, OTHER_BEAM_STYLE,
                            ordered_beam_types, style_summary_axes)
from utils.si_efficiency import MIN_EVENTS_FOR_SCAN, load_si_and_hodo, run_event_count

OUTPUT_DIR = ensure_output_dir("deteff_scan")

# Plot style, after the CMS efficiency plots (see utils.plotting's summary-plot
# style): points with error bars, no connecting lines. The beam-type summary
# uses one colour + marker per detector, in the same black/blue/red scheme the
# per-detector energy plots use for beam types. The hodoscope is plotted
# fiducial (hodo_fid; see module docstring).
STATIONS = ("si1", "si2", "hodo_fid")
STATION_STYLE = {
    "si1": dict(color="#000000", marker="o"),
    "si2": dict(color="#1f4fd1", marker="^"),
    "hodo_fid": dict(color="#d62020", marker="s"),
}
STATION_LABELS = {"si1": "Si Tracker 1", "si2": "Si Tracker 2", "hodo_fid": "Hodoscope (fiducial)"}
# Si1 and Si2 often sit on top of each other: a small sideways nudge per category
# keeps every point visible (beam types at one energy use ENERGY_NUDGE_GEV).
CATEGORY_NUDGE = {"si1": -0.15, "si2": 0.0, "hodo_fid": 0.15}

# One worker per available CPU (respects a SLURM/cgroup allocation, unlike
# os.cpu_count()); bump your job's cpu allocation and this picks it up
# automatically, no flag needed.
N_WORKERS = len(os.sched_getaffinity(0))


def _load_one(run_id, pid=False):
    """Picklable per-run worker for the process pool. With ``pid``, the
    reference is the run's own nominal beam particle (see module docstring)."""
    try:
        pid_beam_type = get_run_beam(run_id)[0] if pid else None
        return run_id, load_si_and_hodo(str(run_id), pid_beam_type=pid_beam_type), None
    except Exception as e:
        return run_id, None, str(e)


def filter_scannable_runs(run_ids):
    """Drop runs too short to possibly align before the expensive load step.

    A handful of "runs" in run_list.json are short calibration/threshold-scan
    blips (tens of events) tagged with a physics beam_type/energy anyway --
    too short to align or measure an efficiency on meaningfully. Filtering
    by event count (cheap: metadata-only) up front avoids paying for the
    full tracker load + alignment on them, and -- more importantly --
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


def load_all_runs(run_ids, pid=False):
    """Load every run in ``run_ids`` in parallel (PID-selected if ``pid``).

    Returns ``(results, skipped)``: ``results`` is ``{run_id: result}`` for
    runs that loaded successfully; ``skipped`` is ``{run_id: reason}`` for
    the rest (bad alignment, missing tracker files, etc.).
    """
    results = {}
    skipped = {}
    with tqdm(total=len(run_ids), desc="Scanning runs", unit="run") as pbar, \
        ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_one, run_id, pid): run_id for run_id in run_ids}
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

    Returns ``(n_si1, n_si2, n_si_events, n_hodo_good, n_events, n_fid_events,
    n_hodo_good_fid, n_hodo_runs, n_si_runs)`` or ``None`` if none of the
    runs in the group loaded successfully.

    The veto-referenced hodoscope counts are summed over every loaded run
    regardless of whether its tracker aligned (``load_si_and_hodo`` returns
    valid n_hodo_good/n_events straight from ROOT even when tracker_error is
    set) -- ``n_hodo_runs`` is how many runs that covers. The si and
    fiducial hodoscope counts only get a real contribution from runs whose
    tracker did align (a tracker_error run contributes zeros), so they're
    unaffected by tracker-unavailable runs; ``n_si_runs`` is that smaller
    count.
    """
    parts = [results_by_run[r] for r in run_ids if r in results_by_run]
    if not parts:
        return None
    counts = tuple(sum(c) for c in zip(*(p[:-1] for p in parts)))
    n_si_runs = sum(1 for p in parts if p[-1] is None)
    return counts + (len(parts), n_si_runs)


# 1-sigma equivalent, matching the previous normal-approximation's implicit
# confidence level (sqrt(p(1-p)/n) is a 1-sigma standard error).
_CL = 0.6827


def _binomial_efficiency(n_pass, n_total):
    """Plain event-count ratio (numerator = veto-passing events with a hit
    on the detector, denominator = all veto-passing events) -- no binning or
    geometric-footprint masking.

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


def station_efficiencies(n_si1, n_si2, n_si_events):
    eff1_mean, eff1_unc = _binomial_efficiency(n_si1, n_si_events)
    eff2_mean, eff2_unc = _binomial_efficiency(n_si2, n_si_events)
    return eff1_mean, eff1_unc, eff2_mean, eff2_unc


def hodo_efficiency(n_hodo_good, n_events):
    return _binomial_efficiency(n_hodo_good, n_events)


def _ordered_beam_types(rows):
    return ordered_beam_types(r["beam_type"] for r in rows)


def _plot_station(ax, x, rows, station_key, label=None, style=None):
    ax.errorbar(x, [r[f"eff_{station_key}"] for r in rows],
                yerr=[r[f"eff_{station_key}_unc"] or 0 for r in rows],
                ls="none", ms=9, elinewidth=1.5, capsize=0, label=label, **(style or STATION_STYLE[station_key]))


def _style_efficiency_axes(ax):
    ax.set_ylim(0, 1.05)
    style_summary_axes(ax)


def plot_eff_vs_beamtype(rows, filename, pid=False):
    """Efficiency per beam type (pooled over energy), one point per detector;
    ``pid`` only labels the plot as PID-tagged."""
    if not rows:
        print("No beam-type rows to plot.")
        return
    by_type = {r["beam_type"]: r for r in rows}
    beam_types = _ordered_beam_types(rows)
    ordered = [by_type[b] for b in beam_types]
    x = np.arange(len(beam_types))

    plt.style.use(mh.style.ROOT)
    fig, ax = plt.subplots(figsize=(10, 8))
    for station_key in STATIONS:
        _plot_station(ax, x + CATEGORY_NUDGE[station_key], ordered, station_key, STATION_LABELS[station_key])
    ax.set_xticks(x)
    ax.set_xticklabels([BEAM_LABELS.get(b, b) for b in beam_types], fontsize=26)
    ax.set_xlim(-0.6, len(beam_types) - 0.4)
    ax.set_xlabel("Beam particle", loc="right")
    ax.set_ylabel("Efficiency", loc="top")
    _style_efficiency_axes(ax)
    ax.xaxis.set_minor_locator(plt.NullLocator())  # categories: no minor ticks or gridlines between them
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    mh.label.exp_label(exp="CaloX", text="Preliminary", data=True, rlabel="PID-tagged" if pid else "", ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Efficiency-vs-beam-type plot saved {filename}")
    plt.close()


def plot_eff_vs_energy(rows, output_dir, pid=False):
    """Efficiency vs. beam energy, one figure per detector
    (eff_vs_energy_<si1|si2|hodo>[_pid].png), with the beam types overlaid."""
    if not rows:
        print("No energy rows to plot.")
        return
    beam_types = _ordered_beam_types(rows)

    plt.style.use(mh.style.ROOT)
    for station_key in STATIONS:
        fig, ax = plt.subplots(figsize=(10, 8))
        for beam_type in beam_types:
            sub = sorted((r for r in rows if r["beam_type"] == beam_type), key=lambda r: r["beam_energy_gev"])
            energies = np.array([r["beam_energy_gev"] for r in sub], dtype=float)
            _plot_station(ax, energies + ENERGY_NUDGE_GEV.get(beam_type, 0.0), sub, station_key,
                          label=BEAM_LABELS.get(beam_type, beam_type), style=BEAM_STYLE.get(beam_type, OTHER_BEAM_STYLE))
        ax.set_xlim(0, 180)
        ax.set_xlabel("Beam energy [GeV]", loc="right")
        ax.set_ylabel("Efficiency", loc="top")
        _style_efficiency_axes(ax)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
        rlabel = STATION_LABELS[station_key] + (", PID" if pid else "")
        mh.label.exp_label(exp="CaloX", text="Preliminary", data=True, rlabel=rlabel, ax=ax)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"eff_vs_energy_{station_key}{'_pid' if pid else ''}.png")
        plt.savefig(filename, dpi=150)
        print(f"Efficiency-vs-energy plot saved {filename}")
        plt.close(fig)


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", action="store_true",
                        help="Only count events the beam-line PID tags as each run's nominal beam particle.")
    args = parser.parse_args()
    suffix = "_pid" if args.pid else ""

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
    results_by_run, skipped_load = load_all_runs(scannable_run_ids, pid=args.pid)

    skipped_all = {**skipped_short, **skipped_load}
    if skipped_all:
        skip_log_path = os.path.join(OUTPUT_DIR, f"skipped_runs{suffix}.log")
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
            (n_si1, n_si2, n_si_events, n_hodo_good, n_events, n_fid_events, n_hodo_good_fid,
             n_hodo_runs, n_si_runs) = result
            pooled_parts.append(result[:-2])

            eff1_mean, eff1_unc, eff2_mean, eff2_unc = station_efficiencies(n_si1, n_si2, n_si_events)
            hodo_eff, hodo_unc = hodo_efficiency(n_hodo_good, n_events)
            fid_eff, fid_unc = hodo_efficiency(n_hodo_good_fid, n_fid_events)
            print(f"  {energy} GeV: n_hodo_runs={n_hodo_runs}/{len(run_ids)}  n_si_runs={n_si_runs}  "
                  f"n_si_events={n_si_events}  n_events={n_events}  n_fid_events={n_fid_events}  "
                  f"si1={eff1_mean:.3f}±{(eff1_unc or 0):.3f}  "
                  f"si2={eff2_mean:.3f}±{(eff2_unc or 0):.3f}  "
                  f"hodo={hodo_eff:.3f}±{(hodo_unc or 0):.3f}  "
                  f"hodo_fid={fid_eff:.3f}±{(fid_unc or 0):.3f}")

            energy_rows.append({
                "beam_type": beam_type, "beam_energy_gev": energy,
                "n_candidate_runs": len(run_ids), "n_hodo_runs": n_hodo_runs, "n_si_runs": n_si_runs,
                "n_si_events": n_si_events, "n_events": n_events, "n_fid_events": n_fid_events,
                "eff_si1": eff1_mean, "eff_si1_unc": eff1_unc,
                "eff_si2": eff2_mean, "eff_si2_unc": eff2_unc,
                "eff_hodo": hodo_eff, "eff_hodo_unc": hodo_unc,
                "eff_hodo_fid": fid_eff, "eff_hodo_fid_unc": fid_unc,
            })

        if not pooled_parts:
            continue
        (n_si1, n_si2, n_si_events, n_hodo_good_total, n_events_total, n_fid_total,
         n_hodo_good_fid_total) = (sum(c) for c in zip(*pooled_parts))
        eff1_mean, eff1_unc, eff2_mean, eff2_unc = station_efficiencies(n_si1, n_si2, n_si_events)
        hodo_eff, hodo_unc = hodo_efficiency(n_hodo_good_total, n_events_total)
        fid_eff, fid_unc = hodo_efficiency(n_hodo_good_fid_total, n_fid_total)
        print(f"  [{beam_type} all energies] n_si_events={n_si_events}  n_events={n_events_total}  "
              f"n_fid_events={n_fid_total}  "
              f"si1={eff1_mean:.3f}±{(eff1_unc or 0):.3f}  "
              f"si2={eff2_mean:.3f}±{(eff2_unc or 0):.3f}  "
              f"hodo={hodo_eff:.3f}±{(hodo_unc or 0):.3f}  "
              f"hodo_fid={fid_eff:.3f}±{(fid_unc or 0):.3f}")
        beamtype_rows.append({
            "beam_type": beam_type, "n_si_events": n_si_events, "n_events": n_events_total,
            "n_fid_events": n_fid_total,
            "eff_si1": eff1_mean, "eff_si1_unc": eff1_unc,
            "eff_si2": eff2_mean, "eff_si2_unc": eff2_unc,
            "eff_hodo": hodo_eff, "eff_hodo_unc": hodo_unc,
            "eff_hodo_fid": fid_eff, "eff_hodo_fid_unc": fid_unc,
        })

    plot_eff_vs_beamtype(beamtype_rows, os.path.join(OUTPUT_DIR, f"eff_vs_beamtype{suffix}.png"), pid=args.pid)
    plot_eff_vs_energy(energy_rows, OUTPUT_DIR, pid=args.pid)
    write_summary(energy_rows, os.path.join(OUTPUT_DIR, f"summary_by_energy{suffix}.csv"))
    write_summary(beamtype_rows, os.path.join(OUTPUT_DIR, f"summary_by_beamtype{suffix}.csv"))


if __name__ == "__main__":
    main()
