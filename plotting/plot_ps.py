from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np

import matplotlib

# Force non-interactive backend to avoid GUI hangs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    def tqdm(iterable, **kwargs):
        return iterable

from nematics.gro import read_gro_frame
from nematics.paf import compute_global_paf, make_molecular_axes


def collect_s_values(
    gro_dir: Path, nframes: int, natoms_per_mol: int, show_progress: bool
) -> np.ndarray:
    s_values = np.zeros(nframes, dtype=float)
    iterator = range(nframes)
    if show_progress:
        iterator = tqdm(iterator, desc=gro_dir.name, unit="frame")
    for i in iterator:
        frame_path = gro_dir / f"frame{i}.gro"
        frame = read_gro_frame(frame_path, natoms_per_mol=natoms_per_mol)
        axes = make_molecular_axes(frame.positions)
        evals, _ = compute_global_paf(axes.molzax)
        s_values[i] = evals[0]
    return s_values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot P(S) from largest PAF eigenvalue across trajectories."
    )
    parser.add_argument(
        "--data-root",
        default="data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign",
        help="Root directory containing gro, gro 2, gro 3, gro 4",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=1000,
        help="Number of frames to read per trajectory",
    )
    parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Atoms per molecule (default: 38 for 5CB)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )
    parser.add_argument(
        "--out",
        default="reports/p_s_hist.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    gro_dirs = [
        data_root / "gro",
        data_root / "gro 2",
        data_root / "gro 3",
        data_root / "gro 4",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    show_progress = not args.no_progress
    for ax, gro_dir in zip(axes, gro_dirs):
        s_values = collect_s_values(
            gro_dir, args.nframes, args.natoms_per_mol, show_progress
        )
        ax.hist(s_values, bins=args.bins, density=True, alpha=0.8, color="#2c7fb8")
        ax.set_title(gro_dir.name)
        ax.set_xlabel("S (largest eigenvalue)")
        ax.set_ylabel("P(S)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Nematic order parameter distribution", fontsize=14)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
