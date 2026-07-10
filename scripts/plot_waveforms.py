import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.waveforms import subtract_baseline


def main():
    parser = argparse.ArgumentParser(description="Plot baseline-subtracted waveforms")
    parser.add_argument("file", help="Path to ROOT file")
    parser.add_argument("branch", help="Branch name to plot")
    parser.add_argument("-n", "--nevents", type=int, default=10,
                        help="Number of events to plot (default: 10)")
    parser.add_argument("--tree", default="EventTree", help="TTree name (default: EventTree)")
    parser.add_argument("--overlay", action="store_true",
                        help="Overlay all events on one axes instead of subplots")
    args = parser.parse_args()

    with uproot.open(args.file) as f:
        tree = f[args.tree]
        raw = np.stack(tree[args.branch].array(library="np"))

    if args.overlay:
        data = subtract_baseline(raw)        # use all events for the 2D histogram
    else:
        n = min(args.nevents, len(raw))
        data = subtract_baseline(raw[:n])

    samples = np.arange(data.shape[1])

    if args.overlay:
        amp_min, amp_max = data.min(), data.max()
        h, xedges, yedges = np.histogram2d(
            np.tile(samples, len(data)),
            data.ravel(),
            bins=[data.shape[1], 300],
            range=[[0, data.shape[1]], [amp_min, amp_max]],
        )
        h = np.ma.masked_where(h == 0, h)

        fig, ax = plt.subplots(figsize=(10, 5))
        mesh = ax.pcolormesh(xedges, yedges, h.T,
                             norm=plt.matplotlib.colors.LogNorm(vmin=1),
                             cmap="jet")
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label("Events")
        ax.set_xlabel("Time Slice")
        ax.set_ylabel("Amplitude (ADC)")
        ax.set_title(f"{args.branch}  |  {os.path.basename(args.file)}")
    else:
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i in range(n):
            ax = axes[i // ncols][i % ncols]
            ax.plot(samples, data[i], lw=0.9)
            ax.set_title(f"Event {i}", fontsize=9)
            ax.set_xlabel("Sample", fontsize=8)
            ax.set_ylabel("ADC", fontsize=8)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"{args.branch}  |  {os.path.basename(args.file)}")

    plt.tight_layout()
    os.makedirs("/lustre/work/colnunn/detreco/output/plot_waveforms", exist_ok=True)
    plt.savefig("/lustre/work/colnunn/detreco/output/plot_waveforms/waveforms.png", dpi=300)
    print("Waveform plot saved to /lustre/work/colnunn/detreco/output/plot_waveforms/waveforms.png")


if __name__ == "__main__":
    main()
