#!/usr/bin/env python3
"""Plot CB7CB biphenyl-spacer junction-dihedral distributions."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


R_KCAL_PER_MOL_K = 0.00198720425864083


def temperature_from_label(label: str) -> int:
    matches = re.findall(r"\d{3}", label)
    if not matches:
        raise ValueError(f"No three-digit temperature code found in run label: {label}")
    return int(matches[-1])


def probability_to_pmf(probability: np.ndarray, temperature: int) -> np.ndarray:
    pmf = np.full(probability.shape, np.nan, dtype=float)
    positive = probability > 0.0
    pmf[positive] = -R_KCAL_PER_MOL_K * temperature * np.log(probability[positive])
    pmf[positive] -= np.min(pmf[positive])
    return pmf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    colors = ["#6f2dbd", "#008b8b", "#e76f51", "#264653"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.0),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    definitions = (
        (
            "left",
            "P_chi_left.csv",
            "Left biphenyl-C29 junction",
            "14-18-29-30 (central bond 18-29)",
        ),
        (
            "right",
            "P_chi_right.csv",
            "Right biphenyl-C35 junction",
            "11-4-35-34 (central bond 4-35)",
        ),
    )
    for column, (side, filename, title, atom_ids) in enumerate(definitions):
        probability_axis = axes[0, column]
        pmf_axis = axes[1, column]
        for index, (label, directory) in enumerate(args.run_dir):
            directory_path = Path(directory)
            data = np.loadtxt(directory_path / filename, delimiter=",", skiprows=1)
            temperature = temperature_from_label(label)
            pmf = probability_to_pmf(data[:, 3], temperature)
            color = colors[index % len(colors)]
            curve_label = f"{label} ({temperature} K)"
            probability_axis.plot(
                data[:, 2], data[:, 3], linewidth=1.8, color=color, label=curve_label
            )
            finite = np.isfinite(pmf)
            pmf_axis.plot(
                data[finite, 2],
                pmf[finite],
                linewidth=1.8,
                color=color,
                label=curve_label,
            )
            np.savetxt(
                directory_path / f"PMF_chi_{side}.csv",
                np.column_stack(
                    (
                        data[:, 2],
                        data[:, 3],
                        pmf,
                        np.full(data.shape[0], temperature),
                    )
                ),
                delimiter=",",
                header="chi_center_deg,P_chi_per_degree,PMF_kcal_per_mol,temperature_K",
                comments="",
                fmt=("%.10f", "%.10f", "%.10f", "%.0f"),
            )

        for axis in (probability_axis, pmf_axis):
            for reference in (0.0, 90.0, 180.0, 270.0, 360.0):
                axis.axvline(reference, color="0.82", linewidth=0.8, linestyle=":")
            axis.set_xlim(0.0, 360.0)
            axis.set_xticks(np.arange(0, 361, 60))
        probability_axis.set_title(f"{title}\nlocal atoms {atom_ids}")
        pmf_axis.axhline(0.0, color="0.75", linewidth=0.9, linestyle=":")
        pmf_axis.set_xlabel(r"$\chi_{\mathrm{junction}}$ (degrees)")

    axes[0, 0].set_ylabel(r"$P(\chi)$ (degree$^{-1}$)")
    axes[1, 0].set_ylabel("PMF (kcal/mol)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "CB7CB biphenyl-spacer junction dihedrals\n"
        r"PMF($\chi$) = $-RT\ln P(\chi)+C$; each minimum set to zero",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

