#!/usr/bin/env python3
"""Plot single-system and common-temperature 4OCB/5CB dihedral PMFs."""

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


R_KCAL_PER_MOL_K = 0.00198720425864083
BIN_EDGES_DEG = np.linspace(0.0, 360.0, 361)
BIN_CENTERS_DEG = 0.5 * (BIN_EDGES_DEG[:-1] + BIN_EDGES_DEG[1:])


@dataclass(frozen=True)
class PmfCurve:
    temperature_k: float
    pmf_kcal_per_mol: np.ndarray


def load_pmf(path: Path) -> PmfCurve:
    metadata_path = path.with_name("dihedral.meta.yaml")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    temperature_k = float(metadata["source"]["temperature_k"])

    angles = np.loadtxt(path, comments="#", usecols=range(1, 1001))
    angles = np.mod(angles.ravel(), 360.0)
    counts, _ = np.histogram(angles, bins=BIN_EDGES_DEG)
    probability = counts.astype(np.float64) / float(np.sum(counts))

    pmf = np.full(probability.shape, np.nan, dtype=np.float64)
    occupied = probability > 0.0
    pmf[occupied] = -R_KCAL_PER_MOL_K * temperature_k * np.log(probability[occupied])
    pmf -= np.nanmin(pmf)
    return PmfCurve(temperature_k=temperature_k, pmf_kcal_per_mol=pmf)


def collect_curves(root: Path, name: str) -> list[PmfCurve]:
    paths = sorted(root.glob(f"*/dihedral/{name}/dihedral.dat"))
    if not paths:
        raise FileNotFoundError(f"No {name} dihedral tables found below {root}")
    curves = sorted((load_pmf(path) for path in paths), key=lambda curve: curve.temperature_k)
    temperatures = [curve.temperature_k for curve in curves]
    if len(temperatures) != len(set(temperatures)):
        raise ValueError(f"Duplicate temperatures found for {name}")
    return curves


def configure_axis(ax: plt.Axes, temperature_k: float, maximum: float) -> None:
    ax.set_title(f"{temperature_k:g} K", fontsize=10)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, maximum * 1.05)
    ax.set_xticks((0, 90, 180, 270, 360))
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8)


def definition(system: str, name: str) -> str:
    if system == "5CB":
        return "8–10–15–16; central bond 10–15"
    if name == "ether":
        return "16–15–28–11; central bond 15–28"
    return "10–11–28–15; central bond 11–28"


def display_name(name: str) -> str:
    return "Ether" if name == "ether" else "Ring-junction"


def plot_all_temperatures(
    curves: list[PmfCurve],
    system: str,
    name: str,
    output: Path,
) -> None:
    columns = 3
    rows = math.ceil(len(curves) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 2.75 * rows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    color = "#0072B2" if system == "5CB" else "#D55E00"
    maximum = max(float(np.nanmax(curve.pmf_kcal_per_mol)) for curve in curves)

    for ax, curve in zip(axes_array, curves):
        ax.plot(BIN_CENTERS_DEG, curve.pmf_kcal_per_mol, color=color, linewidth=1.5)
        configure_axis(ax, curve.temperature_k, maximum)
    for ax in axes_array[len(curves) :]:
        ax.set_visible(False)

    figure.suptitle(
        f"{system} {display_name(name)} dihedral PMFs", fontsize=13, y=0.985
    )
    figure.text(
        0.5,
        0.952,
        definition(system, name),
        ha="center",
        va="center",
        fontsize=9,
        color="0.3",
    )
    figure.supxlabel("Dihedral angle (degrees)", fontsize=10, y=0.035)
    figure.supylabel("PMF (kcal/mol)", fontsize=10, x=0.025)
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.91,
        bottom=0.09,
        hspace=0.30,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_common_temperatures(
    curves_4ocb: list[PmfCurve],
    curves_5cb: list[PmfCurve],
    name_4ocb: str,
    output: Path,
) -> None:
    by_temperature_4ocb = {curve.temperature_k: curve for curve in curves_4ocb}
    by_temperature_5cb = {curve.temperature_k: curve for curve in curves_5cb}
    temperatures = sorted(set(by_temperature_4ocb) & set(by_temperature_5cb))
    if not temperatures:
        raise ValueError("4OCB and 5CB have no common temperatures")

    selected = [
        curve
        for temperature in temperatures
        for curve in (by_temperature_4ocb[temperature], by_temperature_5cb[temperature])
    ]
    maximum = max(float(np.nanmax(curve.pmf_kcal_per_mol)) for curve in selected)
    figure, axes = plt.subplots(
        1,
        len(temperatures),
        figsize=(12.0, 4.2),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, temperature in zip(axes_array, temperatures):
        curve_4ocb = by_temperature_4ocb[temperature]
        curve_5cb = by_temperature_5cb[temperature]
        ax.plot(
            BIN_CENTERS_DEG,
            curve_4ocb.pmf_kcal_per_mol,
            color="#D55E00",
            linewidth=1.5,
        )
        ax.scatter(
            BIN_CENTERS_DEG,
            curve_5cb.pmf_kcal_per_mol,
            color="#0072B2",
            marker="o",
            s=5.0,
            linewidths=0.0,
            zorder=3,
        )
        configure_axis(ax, temperature, maximum)

    figure.suptitle(
        f"4OCB {display_name(name_4ocb)} vs 5CB junction PMFs",
        fontsize=13,
        y=0.985,
    )
    figure.text(
        0.5,
        0.895,
        f"4OCB: {definition('4OCB', name_4ocb)}   |   5CB: {definition('5CB', 'ring-junction')}",
        ha="center",
        va="center",
        fontsize=9,
        color="0.3",
    )
    figure.supxlabel("Dihedral angle (degrees)", fontsize=10, y=0.09)
    figure.supylabel("PMF (kcal/mol)", fontsize=10, x=0.02)
    figure.legend(
        handles=(
            Line2D([0], [0], color="#D55E00", linewidth=1.8, label="4OCB"),
            Line2D(
                [0],
                [0],
                color="#0072B2",
                marker="o",
                linestyle="none",
                markersize=4.0,
                label="5CB",
            ),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.76,
        bottom=0.22,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2"),
        help="4OCB processed dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/4ocb_dihedral_pmfs"),
        help="Directory for the five PNG files",
    )
    parser.add_argument(
        "--5cb-root",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7"),
        help="5CB processed dataset root",
    )
    args = parser.parse_args()

    curves_4ocb_ether = collect_curves(args.root, "ether")
    curves_4ocb_junction = collect_curves(args.root, "ring-junction")
    curves_5cb_junction = collect_curves(args.__dict__["5cb_root"], "ring-junction")
    outputs = (
        args.output_dir / "4ocb_ether_pmf_all_temperatures.png",
        args.output_dir / "4ocb_ring_junction_pmf_all_temperatures.png",
        args.output_dir / "5cb_ring_junction_pmf_all_temperatures.png",
        args.output_dir / "common_temperatures_4ocb_ether_vs_5cb_junction_pmf.png",
        args.output_dir / "common_temperatures_4ocb_ring_junction_vs_5cb_junction_pmf.png",
    )
    plot_all_temperatures(curves_4ocb_ether, "4OCB", "ether", outputs[0])
    plot_all_temperatures(curves_4ocb_junction, "4OCB", "ring-junction", outputs[1])
    plot_all_temperatures(curves_5cb_junction, "5CB", "ring-junction", outputs[2])
    plot_common_temperatures(curves_4ocb_ether, curves_5cb_junction, "ether", outputs[3])
    plot_common_temperatures(
        curves_4ocb_junction,
        curves_5cb_junction,
        "ring-junction",
        outputs[4],
    )
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

