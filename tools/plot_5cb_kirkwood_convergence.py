#!/usr/bin/env python3
"""Compare integrated and raw-sum 5CB CN Kirkwood estimators."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class KirkwoodRun:
    run_id: str
    temperature_k: float
    radius_nm: np.ndarray
    gk_integrated: np.ndarray
    gk_raw: np.ndarray
    density_nm3: float
    box_nm: np.ndarray
    frames: int
    molecules: int


def first_box_nm(trajectory: Path) -> np.ndarray:
    """Read only the first ARC header and periodic-box record."""
    with trajectory.open(encoding="utf-8") as handle:
        header = handle.readline()
        box_line = handle.readline()
    if not header or not box_line:
        raise ValueError(f"Incomplete first ARC frame in {trajectory}")
    fields = box_line.split()
    if len(fields) < 6:
        raise ValueError(f"Invalid ARC box record in {trajectory}: {box_line.rstrip()}")
    box_angstrom = np.asarray([float(value) for value in fields[:3]], dtype=float)
    angles = np.asarray([float(value) for value in fields[3:6]], dtype=float)
    if not np.allclose(angles, 90.0, atol=1.0e-6):
        raise ValueError(f"Kirkwood script currently requires an orthorhombic box: {trajectory}")
    return box_angstrom / 10.0


def load_run(output_dir: Path, maximum_radius_nm: float) -> KirkwoodRun:
    metadata_path = output_dir / "corr.meta.yaml"
    radial_path = output_dir / "radial.csv"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    table = np.genfromtxt(radial_path, delimiter=",", names=True)
    required = {"distance_nm", "raw_p1_sum", "g1"}
    columns = set(table.dtype.names or ())
    missing = required - columns
    if missing:
        raise ValueError(f"{radial_path} is missing columns: {sorted(missing)}")

    frames = int(metadata["calculation"]["frames"])
    molecules = int(metadata["calculation"]["molecules_per_frame"])
    trajectory = Path(metadata["inputs"]["trajectory"])
    box_nm = first_box_nm(trajectory)
    unique_image_limit_nm = 0.5 * float(np.min(box_nm))
    if maximum_radius_nm >= unique_image_limit_nm:
        raise ValueError(
            f"Requested R={maximum_radius_nm:g} nm is not below Lmin/2="
            f"{unique_image_limit_nm:g} nm for {trajectory}"
        )

    radius = np.asarray(table["distance_nm"], dtype=float)
    if radius.size < 2:
        raise ValueError(f"Not enough radial bins in {radial_path}")
    dr = float(np.median(np.diff(radius)))
    if not np.allclose(np.diff(radius), dr, rtol=1.0e-7, atol=1.0e-10):
        raise ValueError(f"Nonuniform radial bins in {radial_path}")
    lower = np.maximum(0.0, radius - 0.5 * dr)
    upper = radius + 0.5 * dr

    pair_density_nm3 = (molecules - 1) / float(np.prod(box_nm))
    shell_r2_dr = (upper**3 - lower**3) / 3.0
    integrated_increment = 4.0 * math.pi * pair_density_nm3 * table["g1"] * shell_r2_dr
    gk_integrated = 1.0 + np.cumsum(integrated_increment)
    gk_raw = 1.0 + 2.0 * np.cumsum(table["raw_p1_sum"]) / (molecules * frames)

    selected = radius <= maximum_radius_nm
    return KirkwoodRun(
        run_id=str(metadata["system"]["run_id"]),
        temperature_k=float(metadata["system"]["temperature_k"]),
        radius_nm=radius[selected],
        gk_integrated=gk_integrated[selected],
        gk_raw=gk_raw[selected],
        density_nm3=pair_density_nm3,
        box_nm=box_nm,
        frames=frames,
        molecules=molecules,
    )


def load_runs(root: Path, maximum_radius_nm: float) -> list[KirkwoodRun]:
    output_dirs = sorted(root.glob("*/corr/polar"))
    runs = [load_run(path, maximum_radius_nm) for path in output_dirs]
    runs.sort(key=lambda run: run.temperature_k)
    if not runs:
        raise FileNotFoundError(f"No completed */corr/polar outputs found below {root}")
    temperatures = [run.temperature_k for run in runs]
    if len(temperatures) != len(set(temperatures)):
        raise ValueError("Duplicate temperatures in polar-correlation outputs")
    return runs


def write_values(runs: list[KirkwoodRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "temperature_k",
                "radius_nm",
                "gk_integrated_g1",
                "gk_raw_p1_sum",
                "difference",
                "pair_density_nm-3",
                "box_x_nm",
                "box_y_nm",
                "box_z_nm",
                "frames",
                "molecules",
            ]
        )
        for run in runs:
            for radius, integrated, raw in zip(
                run.radius_nm, run.gk_integrated, run.gk_raw, strict=True
            ):
                writer.writerow(
                    [
                        run.run_id,
                        f"{run.temperature_k:.8g}",
                        f"{radius:.8f}",
                        f"{integrated:.16e}",
                        f"{raw:.16e}",
                        f"{integrated - raw:.16e}",
                        f"{run.density_nm3:.16e}",
                        *(f"{length:.16e}" for length in run.box_nm),
                        run.frames,
                        run.molecules,
                    ]
                )


def plot(
    runs: list[KirkwoodRun],
    output: Path,
    maximum_radius_nm: float,
) -> None:
    columns = 2
    rows = math.ceil(len(runs) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.0, 3.6 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_array = axes.ravel()
    all_values = [values for run in runs for values in (run.gk_integrated, run.gk_raw)]
    minimum = min(float(np.min(values)) for values in all_values)
    maximum = max(float(np.max(values)) for values in all_values)
    span = maximum - minimum
    limits = (minimum - 0.07 * span, maximum + 0.07 * span)
    for axis, run in zip(axes_array, runs, strict=False):
        axis.plot(
            run.radius_nm,
            run.gk_integrated,
            color="red",
            linestyle=(0, (5, 2)),
            linewidth=4.4,
            zorder=2.5,
            label=r"Integrated $g_1^{\mathrm{rad}}$",
        )
        marker_stride = max(1, len(run.radius_nm) // 55)
        axis.scatter(
            run.radius_nm[::marker_stride],
            run.gk_raw[::marker_stride],
            color="blue",
            marker="o",
            s=58.0,
            linewidths=0.0,
            zorder=3,
            label="Raw pair sums",
        )
        if run.temperature_k in {304.0, 320.0}:
            averaging_window = (run.radius_nm > 1.2) & (run.radius_nm < 3.5)
            mean_gk = float(
                np.mean(
                    0.5
                    * (
                        run.gk_integrated[averaging_window]
                        + run.gk_raw[averaging_window]
                    )
                )
            )
            mean_line = axis.axhline(
                mean_gk,
                color="green",
                linewidth=4.0,
                linestyle="-",
                zorder=1.0,
                label=rf"mean $G_K$ = {mean_gk:.3f}",
            )
            axis.legend(
                handles=[mean_line],
                loc="lower right",
                frameon=True,
                framealpha=0.92,
                fontsize=12,
            )
        axis.text(
            0.96,
            0.93,
            f"{run.temperature_k:g} K",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
            zorder=5,
        )
        axis.set_xlim(0.0, maximum_radius_nm)
        axis.set_ylim(*limits)
        axis.set_xlabel("Cumulative cutoff R (nm)", fontsize=14)
        axis.tick_params(axis="both", labelsize=15, width=1.5, length=7, labelbottom=True)
        axis.grid(alpha=0.18, linewidth=0.8)

    for axis in axes_array[len(runs) :]:
        axis.set_visible(False)
    for row in range(rows):
        axes[row, 0].set_ylabel(r"Cumulative scalar $G_K(R)$", fontsize=14)

    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.013),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    figure.suptitle("5CB CN orientational Kirkwood-factor convergence", fontsize=21, y=0.993)
    figure.text(
        0.5,
        0.046,
        "Red dashed: integrated radial g1. Blue circles: raw pair sums.",
        ha="center",
        fontsize=12.5,
        color="0.3",
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.075, hspace=0.36, wspace=0.12)
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
        "--maximum-radius-nm",
        type=float,
        default=3.5,
        help="Largest cumulative cutoff; must be below half the smallest box side",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/5cb_polar_corr/5cb_kirkwood_convergence.png"),
        help="Output PNG",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7/kirkwood/cn-kirkwood-convergence.csv"),
        help="Output table containing both cumulative estimators",
    )
    args = parser.parse_args()
    if args.maximum_radius_nm <= 0.0:
        parser.error("--maximum-radius-nm must be positive")
    runs = load_runs(args.root, args.maximum_radius_nm)
    if len(runs) != 11:
        raise RuntimeError(f"Expected 11 completed temperatures, found {len(runs)}")
    write_values(runs, args.output_data)
    plot(runs, args.output, args.maximum_radius_nm)
    largest_difference = max(
        abs(float(run.gk_integrated[-1] - run.gk_raw[-1])) for run in runs
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_data}")
    print(f"Largest |integrated - raw| at the final bin: {largest_difference:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
