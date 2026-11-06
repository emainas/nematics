#!/usr/bin/env python3
"""Plot framewise P(S) for all 11 AMOEBA 5CB temperatures.

The two molecular-axis definitions are compared using the already saved
framewise largest eigenvalues of Q. No trajectory data are read.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


END_COLOR = "#d95f02"
INERTIA_COLOR = "#1f9e89"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-to-end-data",
        type=Path,
        default=Path("data/amoeba_5cb_end_to_end_order"),
    )
    parser.add_argument(
        "--inertia-data",
        type=Path,
        default=Path("data/amoeba_5cb_inertia_order"),
    )
    parser.add_argument(
        "--out-data",
        type=Path,
        default=Path("data/amoeba_5cb_inertia_order/p_of_s"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/amoeba_5cb_inertia_order/p_of_s_11panel.png"),
    )
    parser.add_argument("--s-min", type=float, default=0.04)
    parser.add_argument("--s-max", type=float, default=0.71)
    parser.add_argument("--grid-points", type=int, default=1341)
    return parser.parse_args()


def summary_rows(path: Path) -> list[dict[str, str]]:
    with (path / "summary.csv").open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["temperature_K"]))
    if len(rows) != 11:
        raise ValueError(f"Expected 11 runs in {path}, found {len(rows)}")
    return rows


def gaussian_kde(samples: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or samples.size < 2 or not np.isfinite(samples).all():
        raise ValueError("P(S) requires a finite one-dimensional sample array")
    sigma = float(samples.std(ddof=1))
    bandwidth = 1.06 * sigma * samples.size ** (-0.2)
    if bandwidth <= 0.0:
        raise ValueError("Cannot estimate P(S) from zero-variance samples")
    scaled = (grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).mean(axis=1)
    density /= bandwidth * np.sqrt(2.0 * np.pi)
    normalization = float(np.trapezoid(density, grid))
    if normalization <= 0.0:
        raise ValueError("Invalid KDE normalization")
    density /= normalization
    return density, bandwidth


def main() -> None:
    args = parse_args()
    end_rows = summary_rows(args.end_to_end_data)
    inertia_rows = summary_rows(args.inertia_data)
    inertia_by_temperature = {
        float(row["temperature_K"]): row for row in inertia_rows
    }
    grid = np.linspace(args.s_min, args.s_max, args.grid_points)

    distributions: list[dict[str, object]] = []
    for end_row in end_rows:
        temperature = float(end_row["temperature_K"])
        inertia_row = inertia_by_temperature.get(temperature)
        if inertia_row is None:
            raise ValueError(f"Missing inertia-axis result at {temperature:g} K")
        end_samples = np.load(
            args.end_to_end_data / end_row["run"] / "S_by_frame.npy"
        )
        inertia_samples = np.load(
            args.inertia_data / inertia_row["run"] / "S_by_frame.npy"
        )
        if end_samples.shape != (1000,) or inertia_samples.shape != (1000,):
            raise ValueError(
                f"Expected 1000 framewise S samples at {temperature:g} K; "
                f"found {end_samples.shape} and {inertia_samples.shape}"
            )
        end_density, end_bandwidth = gaussian_kde(end_samples, grid)
        inertia_density, inertia_bandwidth = gaussian_kde(inertia_samples, grid)
        distributions.append(
            {
                "temperature": temperature,
                "run": end_row["run"],
                "end_samples": end_samples,
                "inertia_samples": inertia_samples,
                "end_density": end_density,
                "inertia_density": inertia_density,
                "end_bandwidth": end_bandwidth,
                "inertia_bandwidth": inertia_bandwidth,
            }
        )

    args.out_data.mkdir(parents=True, exist_ok=True)
    with (args.out_data / "p_of_s_kde.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("temperature_K", "S", "P_end_to_end", "P_inertia"))
        for item in distributions:
            for s_value, end_p, inertia_p in zip(
                grid,
                item["end_density"],
                item["inertia_density"],
                strict=True,
            ):
                writer.writerow(
                    (
                        f"{float(item['temperature']):.0f}",
                        f"{s_value:.6f}",
                        f"{end_p:.10e}",
                        f"{inertia_p:.10e}",
                    )
                )

    with (args.out_data / "kde_metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "temperature_K",
                "frames",
                "mean_S_end_to_end",
                "mean_S_inertia",
                "bandwidth_end_to_end",
                "bandwidth_inertia",
            )
        )
        for item in distributions:
            end_samples = np.asarray(item["end_samples"])
            inertia_samples = np.asarray(item["inertia_samples"])
            writer.writerow(
                (
                    f"{float(item['temperature']):.0f}",
                    end_samples.size,
                    f"{end_samples.mean():.10f}",
                    f"{inertia_samples.mean():.10f}",
                    f"{float(item['end_bandwidth']):.10f}",
                    f"{float(item['inertia_bandwidth']):.10f}",
                )
            )

    maximum = max(
        float(np.max(np.asarray(item[key])))
        for item in distributions
        for key in ("end_density", "inertia_density")
    )
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12.0, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for ax, item in zip(flat_axes, distributions, strict=False):
        end_density = np.asarray(item["end_density"])
        inertia_density = np.asarray(item["inertia_density"])
        end_samples = np.asarray(item["end_samples"])
        inertia_samples = np.asarray(item["inertia_samples"])
        ax.fill_between(grid, end_density, color=END_COLOR, alpha=0.12)
        ax.plot(
            grid,
            end_density,
            color=END_COLOR,
            linewidth=1.8,
            label="End-to-end axis",
        )
        ax.fill_between(grid, inertia_density, color=INERTIA_COLOR, alpha=0.12)
        ax.plot(
            grid,
            inertia_density,
            color=INERTIA_COLOR,
            linewidth=1.8,
            label="Inertia axis",
        )
        ax.axvline(end_samples.mean(), color=END_COLOR, linewidth=0.9, linestyle=":")
        ax.axvline(
            inertia_samples.mean(), color=INERTIA_COLOR, linewidth=0.9, linestyle=":"
        )
        ax.set_title(f"{float(item['temperature']):.0f} K", fontsize=11)
        ax.set_xlim(args.s_min, args.s_max)
        ax.set_ylim(0.0, 1.07 * maximum)
        ax.grid(alpha=0.18)

    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    flat_axes[-1].text(
        0.5,
        0.27,
        "P(S) = <delta(S - S_frame)> over 1000 frames\n"
        "Solid curves: Gaussian KDE; dotted lines: <S>",
        transform=flat_axes[-1].transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    for row_index in range(4):
        axes[row_index, 0].set_ylabel("P(S)")
    for ax in axes[-1, :2]:
        ax.set_xlabel("Framewise nematic order, S")
    fig.suptitle("5CB framewise nematic-order distributions", fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    (args.out_data / "README.md").write_text(
        "# AMOEBA 5CB framewise P(S)\n\n"
        "The 11-panel plot compares distributions of the largest eigenvalue of "
        "the framewise Q tensor for the end-to-end and inertia-axis molecular "
        "definitions. Each distribution contains 1,000 saved frame values; no "
        "trajectory was reread. `p_of_s_kde.csv` contains the plotted density "
        "values on a common S grid. `kde_metadata.csv` records the Gaussian KDE "
        "bandwidths, chosen separately for each temperature and axis definition "
        "using `h = 1.06 sigma N^(-1/5)`. Each density was renormalized to unit "
        "area over the plotted S range. Dotted lines mark the corresponding "
        "frame mean `<S>`.\n"
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_data / 'p_of_s_kde.csv'}")
    print(f"Wrote {args.out_data / 'kde_metadata.csv'}")


if __name__ == "__main__":
    main()
