#!/usr/bin/env python3
"""Plot rank-0 and rank-2 correlation panels for 4OCB and 5CB."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CURVE_STYLES = {
    "parallel_1": ("Parallel R=1 nm", "-", "#0072B2"),
    "parallel_2": ("Parallel R=2 nm", "--", "#E69F00"),
    "parallel_3": ("Parallel R=3 nm", ":", "#009E73"),
    "perpendicular": ("Perpendicular", "-.", "#D55E00"),
    "radial": ("Radial", (0, (5, 1)), "#CC79A7"),
}
SYSTEM_COLORS = {"5CB": "#0072B2", "4OCB": "#D55E00"}
SYSTEM_LINESTYLES = {"5CB": "-", "4OCB": "--"}
DISPLAY_CURVES = ("parallel_1", "parallel_2", "parallel_3", "perpendicular")


@dataclass(frozen=True)
class RunCurves:
    temperature_k: float
    mean_s: float
    curves: dict[str, tuple[np.ndarray, np.ndarray]]


def read_run(rank_dir: Path, rank: int) -> RunCurves:
    metadata = yaml.safe_load((rank_dir / "corr.meta.yaml").read_text(encoding="utf-8"))
    temperature_k = float(metadata["system"]["temperature_k"])
    value_column = "g0" if rank == 0 else "g_rank"
    order_path = rank_dir.parent.parent / "nematic-order" / "nematic-order.csv"
    order = np.genfromtxt(order_path, delimiter=",", names=True)
    mean_s = float(np.mean(order["lambda_1"]))

    parallel = np.genfromtxt(rank_dir / "parallel.csv", delimiter=",", names=True)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for radius in (1, 2, 3):
        selected = np.isclose(parallel["radius_nm"], float(radius))
        curves[f"parallel_{radius}"] = (
            parallel["z_nm"][selected],
            parallel[value_column][selected],
        )

    for name in ("perpendicular", "radial"):
        table = np.genfromtxt(rank_dir / f"{name}.csv", delimiter=",", names=True)
        curves[name] = (table["distance_nm"], table[value_column])
    return RunCurves(temperature_k=temperature_k, mean_s=mean_s, curves=curves)


def load_system(root: Path, rank: int) -> dict[float, RunCurves]:
    runs: dict[float, RunCurves] = {}
    for rank_dir in sorted(root.glob(f"*/corr/rank-{rank}")):
        run = read_run(rank_dir, rank)
        if run.temperature_k in runs:
            raise ValueError(f"Duplicate temperature {run.temperature_k:g} K below {root}")
        runs[run.temperature_k] = run
    if not runs:
        raise FileNotFoundError(f"No rank-{rank} corr outputs found below {root}")
    return runs


def y_limits(run_sets: list[dict[float, RunCurves]], rank: int) -> tuple[float, float]:
    values = [
        run.curves[key][1]
        for runs in run_sets
        for run in runs.values()
        for key in DISPLAY_CURVES
    ]
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    span = maximum - minimum
    pad = 0.05 * span if span else 0.1
    lower = 0.0 if rank == 0 else min(0.0, minimum - pad)
    return lower, maximum + pad


def configure_axis(ax: plt.Axes, temperature_k: float, rank: int, limits: tuple[float, float]) -> None:
    ax.set_title(f"{temperature_k:g} K", fontsize=10)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(*limits)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Distance (nm)", fontsize=9)
    ax.set_ylabel("g0" if rank == 0 else "g2", fontsize=9)


def mirrored_parallel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror saved |z| bins for a symmetric signed-director presentation."""
    return np.concatenate((-x[::-1], x)), np.concatenate((y[::-1], y))


def padded_limits(values: list[np.ndarray], guides: tuple[float, ...] = ()) -> tuple[float, float]:
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    if guides:
        minimum = min(minimum, *guides)
        maximum = max(maximum, *guides)
    span = maximum - minimum
    pad = max(0.08 * span, 0.02)
    return minimum - pad, maximum + pad


def plot_system(
    runs: dict[float, RunCurves],
    system: str,
    rank: int,
    geometry: str,
    output: Path,
) -> None:
    if geometry not in {"parallel", "perpendicular"}:
        raise ValueError(f"Unsupported geometry: {geometry}")
    temperatures = sorted(runs)
    columns = 3
    rows = math.ceil(len(temperatures) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 3.0 * rows),
        sharex=True,
        sharey=geometry == "perpendicular",
    )
    axes_array = np.atleast_1d(axes).ravel()
    perpendicular_values = [runs[temperature].curves["perpendicular"][1] for temperature in temperatures]
    perpendicular_guides = tuple(runs[temperature].mean_s**2 for temperature in temperatures) if rank == 2 else ()
    perpendicular_limits = padded_limits(perpendicular_values, perpendicular_guides)
    if rank == 0:
        perpendicular_limits = (0.0, perpendicular_limits[1])

    for ax, temperature in zip(axes_array, temperatures):
        run = runs[temperature]
        if geometry == "parallel":
            parallel_values: list[np.ndarray] = []
            for key in ("parallel_1", "parallel_2", "parallel_3"):
                x, y = run.curves[key]
                x, y = mirrored_parallel(x, y)
                parallel_values.append(y)
                _, _, radius_color = CURVE_STYLES[key]
                ax.plot(x, y, color=radius_color, linewidth=1.35)

            guides = (run.mean_s**2,) if rank == 2 else ()
            configure_axis(ax, temperature, rank, padded_limits(parallel_values, guides))
            ax.set_xlim(-5.0, 5.0)
            ax.set_xlabel("Signed distance along director (nm)", fontsize=9)
            for radius in (1, 2, 3):
                _, _, radius_color = CURVE_STYLES[f"parallel_{radius}"]
                for position in (-float(radius), float(radius)):
                    ax.axvline(
                        position,
                        color=radius_color,
                        linestyle=(0, (3, 3)),
                        linewidth=2.0,
                        alpha=0.75,
                        zorder=0,
                    )
        else:
            x, y = run.curves["perpendicular"]
            ax.plot(x, y, color=SYSTEM_COLORS[system], linewidth=1.35)
            configure_axis(ax, temperature, rank, perpendicular_limits)
            ax.set_xlabel("Distance perpendicular to director (nm)", fontsize=9)

        if rank == 2:
            ax.axhline(
                run.mean_s**2,
                color=SYSTEM_COLORS[system],
                linestyle=(0, (2, 2)),
                linewidth=1.0,
                alpha=0.8,
            )

    for ax in axes_array[len(temperatures) :]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=SYSTEM_COLORS[system], linewidth=2.0, label=system)]
    if geometry == "parallel":
        handles.extend(
            Line2D([0], [0], color=color, linewidth=1.5, label=label)
            for label, _, color in (
                CURVE_STYLES["parallel_1"],
                CURVE_STYLES["parallel_2"],
                CURVE_STYLES["parallel_3"],
            )
        )
    if rank == 2:
        handles.append(
            Line2D(
                [0],
                [0],
                color=SYSTEM_COLORS[system],
                linestyle=(0, (2, 2)),
                linewidth=1.2,
                label="⟨S⟩²",
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(f"{system} rank-{rank} {geometry} correlations", fontsize=13, y=0.985)
    top = 0.95
    if geometry == "parallel":
        figure.text(
            0.5,
            0.957,
            "Largest cylinder: R0 = 3.0 nm < Lmin/2 ≈ 3.65 nm",
            ha="center",
            va="center",
            fontsize=9,
            color="0.3",
        )
        top = 0.925
    figure.subplots_adjust(left=0.07, right=0.985, top=top, bottom=0.09, hspace=0.28, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_common(
    runs_5cb: dict[float, RunCurves],
    runs_4ocb: dict[float, RunCurves],
    rank: int,
    output: Path,
) -> None:
    temperatures = sorted(set(runs_5cb) & set(runs_4ocb))
    if not temperatures:
        raise ValueError("The two systems have no common temperatures")
    figure, axes = plt.subplots(
        len(temperatures),
        2,
        figsize=(10.0, 3.4 * len(temperatures)),
        sharex="col",
        sharey=False,
    )
    axes_array = np.asarray(axes).reshape(len(temperatures), 2)
    selected_5cb = {temperature: runs_5cb[temperature] for temperature in temperatures}
    selected_4ocb = {temperature: runs_4ocb[temperature] for temperature in temperatures}
    limits = y_limits([selected_5cb, selected_4ocb], rank)

    for row, temperature in enumerate(temperatures):
        parallel_ax, perpendicular_ax = axes_array[row]
        parallel_values: list[np.ndarray] = []
        for system, runs in (("5CB", selected_5cb), ("4OCB", selected_4ocb)):
            for key in ("parallel_1", "parallel_2", "parallel_3"):
                x, y = runs[temperature].curves[key]
                x, y = mirrored_parallel(x, y)
                parallel_values.append(y)
                _, _, radius_color = CURVE_STYLES[key]
                if system == "5CB":
                    parallel_ax.scatter(
                        x,
                        y,
                        color=radius_color,
                        marker="o",
                        s=5.0,
                        linewidths=0.0,
                        zorder=3,
                    )
                else:
                    parallel_ax.plot(
                        x,
                        y,
                        color=radius_color,
                        linestyle="-",
                        linewidth=1.35,
                    )
            x, y = runs[temperature].curves["perpendicular"]
            if system == "5CB":
                perpendicular_ax.scatter(
                    x,
                    y,
                    color=SYSTEM_COLORS[system],
                    marker="o",
                    s=5.0,
                    linewidths=0.0,
                    zorder=3,
                )
            else:
                perpendicular_ax.plot(
                    x,
                    y,
                    color=SYSTEM_COLORS[system],
                    linestyle="-",
                    linewidth=1.35,
                )

        for ax, geometry in (
            (parallel_ax, "Parallel"),
            (perpendicular_ax, "Perpendicular"),
        ):
            configure_axis(ax, temperature, rank, limits)
            ax.set_title(f"{temperature:g} K — {geometry}", fontsize=10)
        parallel_ax.set_xlim(-5.0, 5.0)
        parallel_ax.set_xlabel("Signed distance along director (nm)", fontsize=9)
        perpendicular_ax.set_xlabel("Distance perpendicular to director (nm)", fontsize=9)
        for radius in (1, 2, 3):
            _, _, radius_color = CURVE_STYLES[f"parallel_{radius}"]
            for position in (-float(radius), float(radius)):
                parallel_ax.axvline(
                    position,
                    color=radius_color,
                    linestyle=(0, (3, 3)),
                    linewidth=2.0,
                    alpha=0.75,
                    zorder=0,
                )
        guides = tuple(
            runs[temperature].mean_s**2 for runs in (selected_5cb, selected_4ocb)
        ) if rank == 2 else ()
        parallel_ax.set_ylim(*padded_limits(parallel_values, guides))
        if rank == 2:
            for system, runs in (("5CB", selected_5cb), ("4OCB", selected_4ocb)):
                for ax in (parallel_ax, perpendicular_ax):
                    ax.axhline(
                        runs[temperature].mean_s**2,
                        color=SYSTEM_COLORS[system],
                        linestyle=(0, (2, 2)),
                        linewidth=1.0,
                        alpha=0.8,
                    )

    system_handles = [
        Line2D(
            [0],
            [0],
            color=SYSTEM_COLORS["5CB"],
            marker="o",
            linestyle="none",
            markersize=4.0,
            label="5CB",
        ),
        Line2D(
            [0],
            [0],
            color=SYSTEM_COLORS["4OCB"],
            linestyle="-",
            linewidth=2.0,
            label="4OCB",
        ),
    ]
    radius_handles = [
        Line2D([0], [0], color=color, linestyle="-", linewidth=1.5, label=label)
        for label, _, color in (
            CURVE_STYLES["parallel_1"],
            CURVE_STYLES["parallel_2"],
            CURVE_STYLES["parallel_3"],
        )
    ]
    guide_handles = []
    if rank == 2:
        guide_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                linestyle=(0, (2, 2)),
                linewidth=1.2,
                label="⟨S⟩² guide",
            )
        )
    figure.legend(
        handles=system_handles + radius_handles + guide_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6 if rank == 2 else 5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(f"Common-temperature rank-{rank} correlations", fontsize=13, y=0.982)
    figure.text(
        0.5,
        0.945,
        "Largest cylinder: R0 = 3.0 nm < Lmin/2 ≈ 3.65 nm",
        ha="center",
        va="center",
        fontsize=9,
        color="0.3",
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.10, hspace=0.30, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--5cb",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7"),
        help="5CB dataset root",
    )
    parser.add_argument(
        "--4ocb",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2"),
        help="4OCB dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/corr"),
        help="Directory for the ten PNG files",
    )
    args = parser.parse_args()

    runs: dict[tuple[str, int], dict[float, RunCurves]] = {}
    for system, root in (("5CB", args.__dict__["5cb"]), ("4OCB", args.__dict__["4ocb"])):
        for rank in (0, 2):
            runs[(system, rank)] = load_system(root, rank)

    outputs = [
        args.output_dir / "5cb_rank0_parallel_corr.png",
        args.output_dir / "5cb_rank0_perpendicular_corr.png",
        args.output_dir / "5cb_rank2_parallel_corr.png",
        args.output_dir / "5cb_rank2_perpendicular_corr.png",
        args.output_dir / "4ocb_rank0_parallel_corr.png",
        args.output_dir / "4ocb_rank0_perpendicular_corr.png",
        args.output_dir / "4ocb_rank2_parallel_corr.png",
        args.output_dir / "4ocb_rank2_perpendicular_corr.png",
        args.output_dir / "common_temperatures_rank0_corr.png",
        args.output_dir / "common_temperatures_rank2_corr.png",
    ]
    plot_system(runs[("5CB", 0)], "5CB", 0, "parallel", outputs[0])
    plot_system(runs[("5CB", 0)], "5CB", 0, "perpendicular", outputs[1])
    plot_system(runs[("5CB", 2)], "5CB", 2, "parallel", outputs[2])
    plot_system(runs[("5CB", 2)], "5CB", 2, "perpendicular", outputs[3])
    plot_system(runs[("4OCB", 0)], "4OCB", 0, "parallel", outputs[4])
    plot_system(runs[("4OCB", 0)], "4OCB", 0, "perpendicular", outputs[5])
    plot_system(runs[("4OCB", 2)], "4OCB", 2, "parallel", outputs[6])
    plot_system(runs[("4OCB", 2)], "4OCB", 2, "perpendicular", outputs[7])
    plot_common(runs[("5CB", 0)], runs[("4OCB", 0)], 0, outputs[8])
    plot_common(runs[("5CB", 2)], runs[("4OCB", 2)], 2, outputs[9])
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
