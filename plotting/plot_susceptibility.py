from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

# Non-interactive backend for scripts
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nematics.susceptibility import compute_susceptibility_timeseries, save_susceptibility_timeseries


def plot_susceptibility_timeseries(
    chi: np.ndarray,
    out_path: Path,
    title: str = "Orientational Susceptibility vs Frame",
) -> float:
    style_path = Path(__file__).with_name("prl.mplstyle")
    if style_path.exists():
        plt.style.use(style_path)

    frames = np.arange(chi.size)
    mean_chi = float(np.mean(chi))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(frames, chi, lw=1.2, color="tab:blue", label="chi(t)")
    ax.axhline(mean_chi, lw=1.5, ls="--", color="black", label=f"mean = {mean_chi:.4f}")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Susceptibility")
    ax.set_title(title)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"Wrote {out_path}")

    return mean_chi


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute and plot susceptibility timeseries from trajectory or saved .npy"
    )
    parser.add_argument("--gro-dir", default=None, help="Directory containing frame*.gro files")
    parser.add_argument("--chi-file", default=None, help="Optional precomputed .npy file")
    parser.add_argument("--out", default="reports/susceptibility_timeseries.png", help="Output plot PNG")
    parser.add_argument("--save-chi", default=None, help="Optional output .npy path for computed chi")
    parser.add_argument("--nframes", type=int, default=1000, help="Number of frames to process")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--natoms-per-mol", type=int, default=38, help="Atoms per molecule")
    parser.add_argument("--axis", choices=["molzax", "ddir"], default="molzax", help="Orientation axis")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display")
    args = parser.parse_args()

    if bool(args.gro_dir) == bool(args.chi_file):
        raise SystemExit("Provide exactly one of --gro-dir or --chi-file")

    if args.chi_file:
        chi = np.load(args.chi_file)
    else:
        chi = compute_susceptibility_timeseries(
            gro_dir=args.gro_dir,
            nframes=args.nframes,
            start_frame=args.start_frame,
            natoms_per_mol=args.natoms_per_mol,
            axis=args.axis,
            show_progress=not args.no_progress,
        )
        if args.save_chi:
            out = save_susceptibility_timeseries(chi, args.save_chi)
            print(f"Wrote {out}")

    mean_chi = plot_susceptibility_timeseries(
        chi,
        out_path=Path(args.out),
    )
    print(f"Mean susceptibility: {mean_chi:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
