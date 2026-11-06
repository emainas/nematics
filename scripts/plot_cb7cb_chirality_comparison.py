#!/usr/bin/env python3
"""Compare pooled CB7CB f(q) distributions from cpptraj matrices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "Q_DATA"),
        required=True,
        help="Label and cpptraj q_by_molecule.dat path; repeat for each run",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.run) < 2:
        raise SystemExit("Provide at least two --run entries")

    edges = np.linspace(-1.0, 1.0, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(
        1,
        len(args.run),
        figsize=(6.2 * len(args.run), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, (label, data_name) in enumerate(args.run):
        ax = axes[index]
        data_path = Path(data_name)
        matrix = np.loadtxt(data_path, comments="#", ndmin=2)
        q = matrix[:, 1:].ravel()
        c_mean = float(q.mean())
        density, _ = np.histogram(q, bins=edges, density=True)
        asymmetry = density - density[::-1]
        ax.plot(centers, density, color="#6f2dbd", linewidth=2.1, label="f(q)")
        ax.fill_between(centers, density, color="#9d4edd", alpha=0.16)
        ax.plot(
            centers,
            asymmetry,
            color="#009688",
            linewidth=1.8,
            label="f(q) - f(-q)",
        )
        ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")
        ax.axvline(0.0, color="0.55", linewidth=1.0, linestyle=":")
        ax.axvline(c_mean, color="#d62828", linewidth=1.5, linestyle="--", label="c")
        ax.text(
            0.96,
            0.94,
            f"c = {c_mean:+.6f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Molecular chiral shape q")
        ax.set_title(label)
        if index == 0:
            ax.set_ylabel("Probability density")
            ax.legend(frameon=False, loc="upper left")

    fig.suptitle("CB7CB induced chiral asymmetry distributions", fontsize=16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
