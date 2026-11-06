#!/usr/bin/env python3
"""Calculate and plot the molecular CB7CB chiral-shape distribution.

For every molecule i and trajectory frame t, cpptraj evaluates

    q_i(t) = (unit(L_i) cross unit(L'_i)) dot unit(d_i,LL')

The script saves the complete cpptraj matrix, pools all molecules and frames
to estimate f(q), and reports c = mean_{i,t}(q_i(t)).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ATOMS_PER_MOLECULE = 65
L_BASE = 18
L_TIP = 7
LP_BASE = 4
LP_TIP = 8
BIPHENYL_1_CARBONS = (1, 2, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26)
BIPHENYL_2_CARBONS = (4, 5, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28)


def global_atom(molecule: int, local_atom: int) -> int:
    return (molecule - 1) * ATOMS_PER_MOLECULE + local_atom


def atom_mask(molecule: int, local_atoms: tuple[int, ...]) -> str:
    return "@" + ",".join(str(global_atom(molecule, atom)) for atom in local_atoms)


def cpptraj_quote(path: Path) -> str:
    value = str(path)
    if '"' in value:
        raise ValueError(f"Double quotes are unsupported in cpptraj paths: {path}")
    return f'"{value}"'


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    output_data: Path,
    molecules: int,
    start: int,
    stop: int,
    stride: int,
) -> None:
    lines = [
        f"parm {cpptraj_quote(topology)}",
        f"trajin {cpptraj_quote(trajectory)} {start} {stop} {stride}",
        # Reconstruct any molecule split across a periodic boundary before
        # calculating its mesogenic centers and directions.
        "autoimage",
    ]

    for molecule in range(1, molecules + 1):
        prefix = f"m{molecule:04d}"
        lines.extend(
            [
                (
                    f"vector {prefix}_L mask @{global_atom(molecule, L_BASE)} "
                    f"@{global_atom(molecule, L_TIP)}"
                ),
                (
                    f"vector {prefix}_Lp mask @{global_atom(molecule, LP_BASE)} "
                    f"@{global_atom(molecule, LP_TIP)}"
                ),
                (
                    f"vector {prefix}_dLL mask "
                    f"{atom_mask(molecule, BIPHENYL_1_CARBONS)} "
                    f"{atom_mask(molecule, BIPHENYL_2_CARBONS)}"
                ),
            ]
        )

    lines.append("run")
    q_datasets: list[str] = []
    for molecule in range(1, molecules + 1):
        prefix = f"m{molecule:04d}"
        q_datasets.append(f"{prefix}_q")
        lines.extend(
            [
                (
                    f"vectormath vec1 {prefix}_L vec2 {prefix}_Lp crossproduct norm "
                    f"name {prefix}_LcrossLp"
                ),
                # With 'norm', cpptraj normalizes both operands in place. The
                # self-dot is retained as a transparent normalization step.
                (
                    f"vectormath vec1 {prefix}_dLL vec2 {prefix}_dLL dotproduct norm "
                    f"name {prefix}_dLL_normcheck"
                ),
                (
                    f"vectormath vec1 {prefix}_LcrossLp vec2 {prefix}_dLL dotproduct "
                    f"name {prefix}_q"
                ),
            ]
        )

    lines.extend(
        [
            "runanalysis",
            (
                f"writedata {cpptraj_quote(output_data)} "
                + " ".join(q_datasets)
                + " prec 12.8"
            ),
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def load_cpptraj_matrix(path: Path, molecules: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.loadtxt(path, comments="#", ndmin=2)
    expected_columns = molecules + 1
    if matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {path}, found {matrix.shape[1]}"
        )
    frames = matrix[:, 0].astype(int)
    q = matrix[:, 1:]
    if not np.isfinite(q).all():
        raise ValueError(f"Non-finite q value found in {path}")
    if np.max(np.abs(q)) > 1.000001:
        raise ValueError(f"q outside [-1, 1] found in {path}")
    return frames, q


def write_derived_data(
    out_dir: Path,
    frames: np.ndarray,
    q: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pooled = q.ravel()
    c_mean = float(pooled.mean())
    edges = np.linspace(-1.0, 1.0, bins + 1)
    density, edges = np.histogram(pooled, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    asymmetry = density - density[::-1]

    np.savetxt(
        out_dir / "c_by_frame.dat",
        np.column_stack((frames, q.mean(axis=1))),
        header="frame c_frame=mean_i(q_i)",
        fmt=("%8d", "% .10f"),
    )
    np.savetxt(
        out_dir / "f_q_histogram.csv",
        np.column_stack((edges[:-1], edges[1:], centers, density, asymmetry)),
        delimiter=",",
        header="q_left,q_right,q_center,f_q,f_q_minus_f_minus_q",
        comments="",
        fmt="%.10f",
    )
    return centers, density, asymmetry, c_mean


def plot_distribution(
    path: Path,
    label: str,
    centers: np.ndarray,
    density: np.ndarray,
    asymmetry: np.ndarray,
    c_mean: float,
    n_frames: int,
    n_molecules: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(centers, density, color="#6f2dbd", linewidth=2.2, label="f(q)")
    ax.fill_between(centers, density, color="#9d4edd", alpha=0.20)
    ax.plot(
        centers,
        asymmetry,
        color="#009688",
        linewidth=1.9,
        label="f(q) - f(-q)",
    )
    ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")
    ax.axvline(0.0, color="0.55", linewidth=1.0, linestyle=":")
    ax.axvline(c_mean, color="#d62828", linewidth=1.8, linestyle="--", label="c")
    ax.text(
        0.97,
        0.95,
        (
            f"c = <q> = {c_mean:+.6f}\n"
            f"{n_molecules} molecules x {n_frames} frames\n"
            f"{n_molecules * n_frames:,} samples"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Molecular chiral shape q")
    ax.set_ylabel("Probability density f(q)")
    ax.set_title(f"CB7CB induced chiral asymmetry: {label}")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--molecules", type=int, default=800)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--cpptraj", default="cpptraj")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    topology = args.topology.resolve()
    trajectory = args.trajectory.resolve()
    out_dir = args.out_dir.resolve()

    if not topology.is_file():
        raise SystemExit(f"Topology not found: {topology}")
    if not trajectory.is_file():
        raise SystemExit(f"Trajectory not found: {trajectory}")
    if shutil.which(args.cpptraj) is None:
        raise SystemExit("cpptraj not found; load Amber 26 first")
    if args.molecules < 1 or args.start < 1 or args.stop < args.start:
        raise SystemExit("Invalid molecule or frame range")
    if args.stride < 1 or args.bins < 2:
        raise SystemExit("Stride must be positive and bins must be at least 2")

    out_dir.mkdir(parents=True, exist_ok=True)
    cpptraj_input = out_dir / "q_all_molecules.cpptraj.in"
    cpptraj_data = out_dir / "q_by_molecule.dat"
    cpptraj_log = out_dir / "q_all_molecules.cpptraj.log"
    write_cpptraj_input(
        cpptraj_input,
        topology,
        trajectory,
        cpptraj_data,
        args.molecules,
        args.start,
        args.stop,
        args.stride,
    )

    with cpptraj_log.open("w") as log:
        subprocess.run(
            [args.cpptraj, "-i", str(cpptraj_input)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    frames, q = load_cpptraj_matrix(cpptraj_data, args.molecules)
    centers, density, asymmetry, c_mean = write_derived_data(
        out_dir, frames, q, args.bins
    )
    plot_distribution(
        out_dir / "f_q.png",
        args.label,
        centers,
        density,
        asymmetry,
        c_mean,
        q.shape[0],
        q.shape[1],
    )

    summary = (
        f"label: {args.label}\n"
        f"trajectory: {trajectory}\n"
        f"frames: {q.shape[0]}\n"
        f"molecules: {q.shape[1]}\n"
        f"samples: {q.size}\n"
        f"c_mean: {c_mean:+.10f}\n"
        f"mean_abs_q: {np.mean(np.abs(q)):.10f}\n"
        f"mean_q_squared: {np.mean(q * q):.10f}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"plot: {out_dir / 'f_q.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
