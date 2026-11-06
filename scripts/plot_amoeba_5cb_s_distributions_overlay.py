#!/usr/bin/env python3
"""Overlay directly binned end-to-end P(S) for all AMOEBA 5CB runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/amoeba_5cb_end_to_end_order"),
    )
    parser.add_argument(
        "--out-data",
        type=Path,
        default=Path(
            "data/amoeba_5cb_end_to_end_order/p_of_s/p_of_s_histogram.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "reports/amoeba_5cb_end_to_end_order/p_of_s_all_temperatures.png"
        ),
    )
    parser.add_argument("--s-min", type=float, default=0.04)
    parser.add_argument("--s-max", type=float, default=0.71)
    parser.add_argument("--bin-width", type=float, default=0.001)
    return parser.parse_args()


def load_runs(data_dir: Path) -> list[tuple[float, str, np.ndarray]]:
    with (data_dir / "summary.csv").open(newline="") as handle:
        summary = sorted(
            csv.DictReader(handle), key=lambda row: float(row["temperature_K"])
        )
    if len(summary) != 11:
        raise ValueError(f"Expected 11 runs in {data_dir}, found {len(summary)}")

    runs: list[tuple[float, str, np.ndarray]] = []
    for row in summary:
        temperature = float(row["temperature_K"])
        run = row["run"]
        samples = np.load(data_dir / run / "S_by_frame.npy")
        if samples.shape != (1000,) or not np.isfinite(samples).all():
            raise ValueError(
                f"Expected 1000 finite framewise S values for {run}, "
                f"found {samples.shape}"
            )
        runs.append((temperature, run, samples))
    return runs


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0.0:
        raise ValueError("--bin-width must be positive")
    span_in_bins = (args.s_max - args.s_min) / args.bin_width
    if not np.isclose(span_in_bins, round(span_in_bins), atol=1.0e-10):
        raise ValueError("The requested S range must be divisible by --bin-width")

    runs = load_runs(args.data_dir)
    nbins = int(round(span_in_bins))
    edges = np.linspace(args.s_min, args.s_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    histograms: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for temperature, run, samples in runs:
        if samples.min() < edges[0] or samples.max() > edges[-1]:
            raise ValueError(
                f"S values for {run} lie outside [{edges[0]}, {edges[-1]}]"
            )
        counts, _ = np.histogram(samples, bins=edges)
        probability = counts / counts.sum()
        density = probability / args.bin_width
        histograms.append((temperature, run, counts, density))

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    with args.out_data.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "temperature_K",
                "run",
                "S_left",
                "S_right",
                "S_center",
                "count",
                "probability",
                "probability_density",
            )
        )
        for temperature, run, counts, density in histograms:
            for left, right, center, count, probability_density in zip(
                edges[:-1], edges[1:], centers, counts, density, strict=True
            ):
                writer.writerow(
                    (
                        f"{temperature:.0f}",
                        run,
                        f"{left:.6f}",
                        f"{right:.6f}",
                        f"{center:.6f}",
                        int(count),
                        f"{count / 1000.0:.8f}",
                        f"{probability_density:.8f}",
                    )
                )

    temperatures = np.asarray([item[0] for item in histograms])
    cmap = plt.get_cmap("turbo", len(temperatures))
    boundaries = np.arange(temperatures.min() - 1.0, temperatures.max() + 3.0, 2.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for index, (temperature, _run, _counts, density) in enumerate(histograms):
        # Do not draw the long zero-density baseline; retain zero-valued bins
        # between occupied bins so genuinely separated modes remain separated.
        occupied = np.flatnonzero(density > 0.0)
        first = max(0, int(occupied[0]) - 1)
        last = min(density.size, int(occupied[-1]) + 2)
        ax.stairs(
            density[first:last],
            edges[first : last + 1],
            color=cmap(index),
            linewidth=1.8,
            label=f"{temperature:.0f} K",
        )

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(
        scalar_map,
        ax=ax,
        ticks=temperatures,
        boundaries=boundaries,
        pad=0.025,
    )
    colorbar.set_label("Temperature (K)")
    ax.set_xlim(args.s_min, args.s_max)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Framewise nematic order, S")
    ax.set_ylabel("P(S)")
    ax.set_title("5CB end-to-end nematic-order distributions")
    ax.grid(alpha=0.20)
    ax.text(
        0.98,
        0.96,
        f"1000 frames per temperature; bin width = {args.bin_width:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_data}")


if __name__ == "__main__":
    main()
