#!/usr/bin/env python3
"""Calculate the two CB7CB biphenyl-spacer junction dihedrals with cpptraj."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np


ATOMS_PER_MOLECULE = 65

# Each quartet is ring-neighbor -> ring-ipso -> first linker C -> second linker C.
# The central bonds are therefore C18-C29 and C4-C35, respectively.
LEFT_DIHEDRAL = (14, 18, 29, 30)
RIGHT_DIHEDRAL = (11, 4, 35, 34)


def global_atom(molecule: int, local_atom: int) -> int:
    return (molecule - 1) * ATOMS_PER_MOLECULE + local_atom


def cpptraj_quote(path: Path) -> str:
    value = str(path)
    if '"' in value:
        raise ValueError(f"Double quotes are unsupported in cpptraj paths: {path}")
    return f'"{value}"'


def dihedral_masks(molecule: int, local_atoms: tuple[int, int, int, int]) -> str:
    return " ".join(f"@{global_atom(molecule, atom)}" for atom in local_atoms)


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    left_data: Path,
    right_data: Path,
    molecules: int,
    start: int,
    stop: int,
    stride: int,
) -> None:
    lines = [
        f"parm {cpptraj_quote(topology)}",
        f"trajin {cpptraj_quote(trajectory)} {start} {stop} {stride}",
        "autoimage",
    ]
    left_names: list[str] = []
    right_names: list[str] = []
    for molecule in range(1, molecules + 1):
        left_name = f"m{molecule:04d}_junction_left"
        right_name = f"m{molecule:04d}_junction_right"
        left_names.append(left_name)
        right_names.append(right_name)
        lines.extend(
            [
                f"dihedral {left_name} {dihedral_masks(molecule, LEFT_DIHEDRAL)}",
                f"dihedral {right_name} {dihedral_masks(molecule, RIGHT_DIHEDRAL)}",
            ]
        )
    lines.extend(
        [
            "run",
            f"writedata {cpptraj_quote(left_data)} {' '.join(left_names)} prec 12.6",
            f"writedata {cpptraj_quote(right_data)} {' '.join(right_names)} prec 12.6",
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def load_matrix(path: Path, molecules: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.loadtxt(path, comments="#", ndmin=2)
    if matrix.shape[1] != molecules + 1:
        raise ValueError(
            f"Expected {molecules + 1} columns in {path}, found {matrix.shape[1]}"
        )
    frames = matrix[:, 0].astype(int)
    angles = matrix[:, 1:]
    if not np.isfinite(angles).all():
        raise ValueError(f"Non-finite dihedral value found in {path}")
    return frames, (angles + 180.0) % 360.0 - 180.0


def write_histogram(path: Path, angles: np.ndarray, bin_width: float) -> None:
    n_bins = round(360.0 / bin_width)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    density, edges = np.histogram(angles.ravel() % 360.0, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(
        path,
        np.column_stack((edges[:-1], edges[1:], centers, density)),
        delimiter=",",
        header="chi_lower_deg,chi_upper_deg,chi_center_deg,P_chi_per_degree",
        comments="",
        fmt="%.10f",
    )


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
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--cpptraj", default="cpptraj")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    topology = args.topology.resolve()
    trajectory = args.trajectory.resolve()
    out_dir = args.out_dir.resolve()
    if not topology.is_file() or not trajectory.is_file():
        raise SystemExit("Topology or trajectory not found")
    if shutil.which(args.cpptraj) is None:
        raise SystemExit("cpptraj not found; load Amber 26 first")
    if args.molecules < 1 or args.start < 1 or args.stop < args.start:
        raise SystemExit("Invalid molecule or frame range")
    if args.stride < 1 or args.bin_width <= 0 or 360 % args.bin_width != 0:
        raise SystemExit("Stride must be positive; bin width must divide 360 degrees")

    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "ring_junction_dihedrals.cpptraj.in"
    log_path = out_dir / "ring_junction_dihedrals.cpptraj.log"
    left_path = out_dir / "chi_left_by_molecule.dat"
    right_path = out_dir / "chi_right_by_molecule.dat"
    write_cpptraj_input(
        input_path,
        topology,
        trajectory,
        left_path,
        right_path,
        args.molecules,
        args.start,
        args.stop,
        args.stride,
    )
    with log_path.open("w") as log:
        subprocess.run(
            [args.cpptraj, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    left_frames, left = load_matrix(left_path, args.molecules)
    right_frames, right = load_matrix(right_path, args.molecules)
    if not np.array_equal(left_frames, right_frames):
        raise ValueError("Left and right cpptraj frame indices differ")
    write_histogram(out_dir / "P_chi_left.csv", left, args.bin_width)
    write_histogram(out_dir / "P_chi_right.csv", right, args.bin_width)
    summary = (
        f"label: {args.label}\n"
        f"trajectory: {trajectory}\n"
        f"frames: {left.shape[0]}\n"
        f"molecules: {left.shape[1]}\n"
        f"samples_per_dihedral: {left.size}\n"
        f"left_local_atoms: {'-'.join(map(str, LEFT_DIHEDRAL))}\n"
        f"right_local_atoms: {'-'.join(map(str, RIGHT_DIHEDRAL))}\n"
        f"left_central_bond: 18-29\n"
        f"right_central_bond: 4-35\n"
        f"histogram_bin_width_deg: {args.bin_width:g}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

