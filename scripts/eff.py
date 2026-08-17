"""Per-run tracker efficiency: X/Y hit rate of each si-tracker plane and the
hodoscope, each plane's X&Y (2D) hit rate, and the tracker/tracker+hodo
track coincidence rate -- all over every ROOT event in the run (no veto or
reference-selection gating).

Usage:
    python -m scripts.eff --run <run_id>
"""

import argparse
import os

from utils.io import ensure_output_dir
from utils.plotting import get_beam_label
from utils.si_efficiency import load_axis_report

OUTPUT_DIR = ensure_output_dir("eff")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()

    try:
        r = load_axis_report(args.run)
    except Exception as e:
        print(f"Error processing run {args.run}: {e}")
        return

    beam_type = get_beam_label(args.run)
    n = r["n_events"]
    x1, y1, x2, y2, hx, hy = r["x1"], r["y1"], r["x2"], r["y2"], r["hodo_x"], r["hodo_y"]

    hit1 = x1 & y1
    hit2 = x2 & y2
    hit_hodo = hx & hy
    track = hit1 & hit2
    track_hodo = track & hit_hodo

    header = f"{'Station':<10}{'X eff':>10}{'Y eff':>10}"
    lines = [
        f"Run {args.run} ({beam_type})",
        header,
        "-" * len(header),
        f"{'1':<10}{x1.sum() / n:>10.3f}{y1.sum() / n:>10.3f}",
        f"{'2':<10}{x2.sum() / n:>10.3f}{y2.sum() / n:>10.3f}",
        f"{'Hodo':<10}{hx.sum() / n:>10.3f}{hy.sum() / n:>10.3f}",
        "-" * len(header),
        f"{'Station':<10}{'X & Y eff':>10}",
        f"{'1':<10}{hit1.sum() / n:>10.3f}",
        f"{'2':<10}{hit2.sum() / n:>10.3f}",
        f"{'Hodo':<10}{hit_hodo.sum() / n:>10.3f}",
        "-" * len(header),
        f"{'Track':<10}{'eff':>10}",
        f"{'Trackers':<10}{track.sum() / n:>10.3f}",
        f"{'Trackers + Hodo':<10}{track_hodo.sum() / n:>10.3f}",
    ]
    print("\n".join(lines))

    out_path = os.path.join(OUTPUT_DIR, f"eff_{args.run}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
