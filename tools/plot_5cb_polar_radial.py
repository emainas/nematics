#!/usr/bin/env python3
"""Plot 5CB CN radial g0, g1, and g2 for every temperature."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PolarRadialRun:
    temperature_k: float
    distance_nm: np.ndarray
    g0: np.ndarray
    g1: np.ndarray
    g2: np.ndarray


def load_runs(root: Path) -> list[PolarRadialRun]:
    runs: list[PolarRadialRun] = []
    for output_dir in sorted(root.glob("*/corr/polar")):
        radial_path = output_dir / "radial.csv"
        metadata_path = output_dir / "corr.meta.yaml"
        if not radial_path.is_file() or not metadata_path.is_file():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        table = np.genfromtxt(radial_path, delimiter=",", names=True)
        required = {"distance_nm", "g0", "g1", "g2"}
        columns = set(table.dtype.names or ())
        missing = required - columns
        if missing:
            raise ValueError(f"{radial_path} is missing columns: {sorted(missing)}")
        runs.append(
            PolarRadialRun(
                temperature_k=float(metadata["system"]["temperature_k"]),
                distance_nm=np.asarray(table["distance_nm"]),
                g0=np.asarray(table["g0"]),
                g1=np.asarray(table["g1"]),
                g2=np.asarray(table["g2"]),
            )
        )
    runs.sort(key=lambda run: run.temperature_k)
    if not runs:
        raise FileNotFoundError(f"No completed */corr/polar outputs found below {root}")
    temperatures = [run.temperature_k for run in runs]
    if len(temperatures) != len(set(temperatures)):
        raise ValueError("Duplicate temperatures found in polar-correlation outputs")
    return runs


def padded_limits(values: list[np.ndarray], include: float) -> tuple[float, float]:
    minimum = min(include, min(float(np.nanmin(value)) for value in values))
    maximum = max(include, max(float(np.nanmax(value)) for value in values))
    span = maximum - minimum
    padding = max(0.07 * span, 0.015)
    return minimum - padding, maximum + padding


def plot(runs: list[PolarRadialRun], output: Path) -> None:
    columns = len(runs)
    figure, axes = plt.subplots(
        3,
        columns,
        figsize=(2.65 * columns, 8.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    colors = {"g0": "#3B6FB6", "g1": "#7A3E9D", "g2": "#D17A22"}
    limits = {
        "g0": padded_limits([run.g0 for run in runs], 1.0),
        "g1": padded_limits([run.g1 for run in runs], 0.0),
        "g2": padded_limits([run.g2 for run in runs], 0.0),
    }

    for column, run in enumerate(runs):
        axes[0, column].set_title(f"{run.temperature_k:g} K", fontsize=10)
        for row, (name, baseline) in enumerate((("g0", 1.0), ("g1", 0.0), ("g2", 0.0))):
            axis = axes[row, column]
            values = getattr(run, name)
            axis.plot(run.distance_nm, values, color=colors[name], linewidth=1.25)
            axis.axhline(baseline, color="0.45", linestyle=(0, (2, 2)), linewidth=0.75)
            axis.set_xlim(0.0, 5.0)
            axis.set_ylim(*limits[name])
            axis.grid(alpha=0.16, linewidth=0.5)
            axis.tick_params(labelsize=7)
        positive = np.maximum(run.g1, 0.0)
        negative = np.minimum(run.g1, 0.0)
        axes[1, column].fill_between(
            run.distance_nm, 0.0, positive, color="#4C78A8", alpha=0.27, linewidth=0.0
        )
        axes[1, column].fill_between(
            run.distance_nm, 0.0, negative, color="#D95F59", alpha=0.27, linewidth=0.0
        )
        axes[2, column].set_xlabel("r (nm)", fontsize=8)

    row_labels = (
        r"$g_0^{\mathrm{rad}}(r)$",
        r"$g_1^{\mathrm{rad}}(r)$",
        r"$g_2^{\mathrm{rad}}(r)$",
    )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11)

    figure.suptitle("5CB CN midpoint/dipole radial correlations", fontsize=14, y=0.987)
    figure.text(
        0.5,
        0.012,
        r"Shading marks the sign of $g_1^{\mathrm{rad}}$; "
        r"$g_K(R)-1=4\pi\rho\int_0^R r^2g_1^{\mathrm{rad}}(r)\,dr$.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.25",
    )
    figure.subplots_adjust(left=0.047, right=0.995, top=0.94, bottom=0.085, hspace=0.15, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7"),
        help="5CB dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/5cb_polar_corr/5cb_polar_radial_all_temperatures.png"),
        help="Output PNG",
    )
    args = parser.parse_args()
    runs = load_runs(args.root)
    if len(runs) != 11:
        raise RuntimeError(f"Expected 11 completed temperatures, found {len(runs)}")
    plot(runs, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
