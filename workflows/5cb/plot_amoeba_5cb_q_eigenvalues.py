#!/usr/bin/env python3
"""Plot all three Q-tensor eigenvalue time series for AMOEBA 5CB runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "reports/amoeba_5cb_end_to_end_order/eigenvalues_timeseries_panel.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.data_dir / "summary.csv").open() as handle:
        rows = sorted(
            csv.DictReader(handle), key=lambda row: float(row["temperature_K"])
        )
    if len(rows) != 11:
        raise ValueError(f"Expected 11 runs, found {len(rows)}")

    colors = ("#0072B2", "#009E73", "#D55E00")
    labels = ("smallest eigenvalue", "middle eigenvalue", "largest eigenvalue = S")
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12.0, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    for ax, row in zip(flat_axes, rows, strict=False):
        run = row["run"]
        temperature = float(row["temperature_K"])
        eigenvalues = np.load(args.data_dir / run / "Q_eigenvalues_by_frame.npy")
        if eigenvalues.shape != (1000, 3):
            raise ValueError(
                f"Expected a (1000, 3) eigenvalue array for {run}, "
                f"found {eigenvalues.shape}"
            )
        frames = np.arange(1, eigenvalues.shape[0] + 1)
        for column, (color, label) in enumerate(zip(colors, labels, strict=True)):
            ax.plot(
                frames,
                eigenvalues[:, column],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                label=label,
            )
        ax.axhline(0.0, color="0.5", linewidth=0.7, linestyle=":", zorder=0)
        mean_trace = float(eigenvalues.sum(axis=1).mean())
        trace_label = (
            "Tr(Q) ~ 0"
            if abs(mean_trace) < 1.0e-10
            else f"Tr(Q) = s1+s2+s3\n= {mean_trace:+.2e} (frame mean)"
        )
        ax.text(
            0.03,
            0.51,
            trace_label,
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88, "pad": 2.5},
        )
        ax.set_title(f"{temperature:.0f} K", fontsize=11)
        ax.grid(alpha=0.18)
        ax.set_xlim(1, 1000)
        ax.set_ylim(-0.40, 0.72)

    flat_axes[-1].axis("off")
    for row_index in range(4):
        axes[row_index, 0].set_ylabel("Q eigenvalue")
    for ax in axes[-1, :2]:
        ax.set_xlabel("Saved frame")

    handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(
        handles,
        legend_labels,
        loc="center",
        frameon=False,
        fontsize=11,
    )
    fig.suptitle("5CB Q-tensor eigenvalues", fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
