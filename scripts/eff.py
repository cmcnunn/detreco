"""Per-run si-tracker efficiency: intrinsic efficiency of each plane (1D),
each whole tracker (2D), and the coincidence between them (3D), referenced
against hodoscope-good + veto-passing events.

Usage:
    python -m scripts.eff --run <run_id>
"""

import argparse
import os

from utils.io import ensure_output_dir
from utils.plotting import get_beam_label
from utils.si_efficiency import load_si_ref_and_hits

OUTPUT_DIR = ensure_output_dir("eff")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the intrinsic efficiency of a detector given a reference and selected hit pattern."
    )
    parser.add_argument("--run", type=str, help="Run ID to process")
    args = parser.parse_args()

    try:
        xh_ref, yh_ref, xh_sel1, yh_sel1, xh_sel2, yh_sel2 = load_si_ref_and_hits(args.run)
    except Exception as e:
        print(f"Error processing run {args.run}: {e}")
        return

    beam_type = get_beam_label(args.run)
    n_ref = len(xh_ref)
    eff1 = len(xh_sel1) / n_ref if n_ref else 0.0
    eff2 = len(xh_sel2) / n_ref if n_ref else 0.0

    lines = [
        f"Run {args.run} ({beam_type})",
        f"n_ref_events = {n_ref}",
        f"{'Station':<10}{'X & Y eff':>10}",
        f"{'1':<10}{eff1:>10.3f}",
        f"{'2':<10}{eff2:>10.3f}",
    ]
    print("\n".join(lines))

    with open(os.path.join(OUTPUT_DIR, f"si_eff_{args.run}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Output saved to {os.path.join(OUTPUT_DIR, f'si_eff_{args.run}.txt')}")


if __name__ == "__main__":
    main()
