from __future__ import annotations

import argparse
from pathlib import Path

from .gro import read_gro_frame
from .order import compute_and_save_order
from .paf import compute_global_paf, make_molecular_axes


def main() -> int:
    parser = argparse.ArgumentParser(description="Nematics analysis tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paf_parser = subparsers.add_parser("paf", help="Compute global PAF from a GRO frame")
    paf_parser.add_argument(
        "--gro-dir",
        required=True,
        help="Directory containing frame*.gro files",
    )
    paf_parser.add_argument("--frame", type=int, default=0, help="Frame index to read")
    paf_parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Number of atoms per molecule (default: 38 for 5CB)",
    )

    order_parser = subparsers.add_parser(
        "order", help="Compute and save S for all trajectories"
    )
    order_parser.add_argument(
        "--data-root",
        default="data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign",
        help="Root directory containing gro, gro 2, gro 3, gro 4",
    )
    order_parser.add_argument(
        "--out-dir",
        default="data/order",
        help="Output directory for .npy files",
    )
    order_parser.add_argument(
        "--nframes",
        type=int,
        default=1000,
        help="Number of frames to read per trajectory",
    )
    order_parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Atoms per molecule (default: 38 for 5CB)",
    )
    order_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    args = parser.parse_args()

    if args.command == "paf":
        gro_path = Path(args.gro_dir) / f"frame{args.frame}.gro"
        frame = read_gro_frame(gro_path, natoms_per_mol=args.natoms_per_mol)
        axes = make_molecular_axes(frame.positions)
        evals, evecs = compute_global_paf(axes.molzax)

        print(f"Frame: {gro_path}")
        print("Eigenvalues (OPs) and eigenvectors of PAF:")
        for i in range(3):
            evec = evecs[:, i]
            print(
                f"{evals[i]:12.6f} {evec[0]:12.6f} {evec[1]:12.6f} {evec[2]:12.6f}"
            )
        return 0

    if args.command == "order":
        outputs = compute_and_save_order(
            data_root=args.data_root,
            out_dir=args.out_dir,
            nframes=args.nframes,
            natoms_per_mol=args.natoms_per_mol,
            show_progress=not args.no_progress,
        )
        for path in outputs:
            print(f"Wrote {path}")
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
