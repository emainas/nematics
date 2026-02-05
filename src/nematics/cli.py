from __future__ import annotations

import argparse
from pathlib import Path

from .gro import read_gro_frame
from .paf import compute_global_paf, make_molecular_axes


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute global PAF from GRO frames.")
    parser.add_argument(
        "--gro-dir",
        required=True,
        help="Directory containing frame*.gro files",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to read")
    parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Number of atoms per molecule (default: 38 for 5CB)",
    )
    args = parser.parse_args()

    gro_path = Path(args.gro_dir) / f"frame{args.frame}.gro"
    frame = read_gro_frame(gro_path, natoms_per_mol=args.natoms_per_mol)
    axes = make_molecular_axes(frame.positions)
    evals, evecs = compute_global_paf(axes.molzax)

    print(f"Frame: {gro_path}")
    print("Eigenvalues (OPs) and eigenvectors of PAF:")
    for i in range(3):
        evec = evecs[:, i]
        print(f"{evals[i]:12.6f} {evec[0]:12.6f} {evec[1]:12.6f} {evec[2]:12.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
