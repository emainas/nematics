from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fortran_backend import build_paircorr_fortran
from .gro import read_gro_frame
from .order import compute_and_save_order
from .paf import compute_global_paf, make_molecular_axes
from .paircorr import compute_paircorr_for_dir, save_paircorr_results
from .susceptibility import compute_susceptibility_timeseries, save_susceptibility_timeseries


def _resolve_input_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return path


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
    paf_parser.add_argument(
        "--axis",
        choices=["molzax", "ddir"],
        default="molzax",
        help="Orientation axis used in global PAF",
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
        "--axis",
        choices=["molzax", "ddir"],
        default="molzax",
        help="Orientation axis used in global PAF. molzax preserves legacy report output.",
    )
    order_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    susc_parser = subparsers.add_parser(
        "susceptibility",
        help="Compute orientational susceptibility timeseries from a trajectory",
    )
    susc_parser.add_argument(
        "--gro-dir",
        required=True,
        help="Directory containing frame*.gro files",
    )
    susc_parser.add_argument(
        "--out",
        default="data/susceptibility/chi.npy",
        help="Output .npy path for susceptibility timeseries",
    )
    susc_parser.add_argument(
        "--nframes",
        type=int,
        default=1000,
        help="Number of frames to read",
    )
    susc_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index",
    )
    susc_parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Atoms per molecule",
    )
    susc_parser.add_argument(
        "--axis",
        choices=["molzax", "ddir"],
        default="molzax",
        help="Orientation axis used for susceptibility",
    )
    susc_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    paircorr_parser = subparsers.add_parser(
        "paircorr",
        help="Compute and save Fortran-style 5CB/CN pair-correlation histograms",
    )
    paircorr_parser.add_argument(
        "--gro-dir",
        required=True,
        help="Directory containing frame*.gro files",
    )
    paircorr_parser.add_argument(
        "--out-dir",
        default="data/paircorr",
        help="Output directory for .dat histogram files",
    )
    paircorr_parser.add_argument(
        "--label",
        default=None,
        help="Output filename prefix (default: gro directory name)",
    )
    paircorr_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index",
    )
    paircorr_parser.add_argument(
        "--nframes",
        type=int,
        default=1000,
        help="Number of frames to read",
    )
    paircorr_parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate hist files every N processed frames (0 disables, faster)",
    )
    paircorr_parser.add_argument(
        "--natoms-per-mol",
        type=int,
        default=38,
        help="Atoms per molecule (default: 38 for 5CB)",
    )
    paircorr_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )
    paircorr_parser.add_argument(
        "--backend",
        choices=["auto", "fortran", "python"],
        default="auto",
        help="Paircorr compute backend",
    )
    paircorr_parser.add_argument(
        "--auto-build-fortran",
        action="store_true",
        help="Automatically compile Fortran backend if missing",
    )

    build_parser = subparsers.add_parser(
        "build-fortran",
        help="Compile Fortran extension backend used by paircorr",
    )
    build_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce build output",
    )

    args = parser.parse_args()

    if args.command == "paf":
        gro_dir = _resolve_input_path(args.gro_dir)
        gro_path = gro_dir / f"frame{args.frame}.gro"
        frame = read_gro_frame(gro_path, natoms_per_mol=args.natoms_per_mol)
        axes = make_molecular_axes(frame.positions)
        dirs = axes.molzax if args.axis == "molzax" else axes.ddir
        evals, evecs = compute_global_paf(dirs)

        print(f"Frame: {gro_path}")
        print(f"Axis: {args.axis}")
        print("Eigenvalues (OPs) and eigenvectors of PAF:")
        for i in range(3):
            evec = evecs[:, i]
            print(
                f"{evals[i]:12.6f} {evec[0]:12.6f} {evec[1]:12.6f} {evec[2]:12.6f}"
            )
        return 0

    if args.command == "order":
        data_root = _resolve_input_path(args.data_root)
        outputs = compute_and_save_order(
            data_root=data_root,
            out_dir=args.out_dir,
            nframes=args.nframes,
            natoms_per_mol=args.natoms_per_mol,
            show_progress=not args.no_progress,
            axis=args.axis,
        )
        for path in outputs:
            print(f"Wrote {path}")
        return 0

    if args.command == "susceptibility":
        gro_dir = _resolve_input_path(args.gro_dir)
        chi = compute_susceptibility_timeseries(
            gro_dir=gro_dir,
            nframes=args.nframes,
            start_frame=args.start_frame,
            natoms_per_mol=args.natoms_per_mol,
            axis=args.axis,
            show_progress=not args.no_progress,
        )
        out = save_susceptibility_timeseries(chi, args.out)
        print(f"Wrote {out}")
        print(f"Mean susceptibility: {float(np.mean(chi)):.6f}")
        return 0

    if args.command == "paircorr":
        gro_dir = _resolve_input_path(args.gro_dir)
        results = compute_paircorr_for_dir(
            gro_dir=gro_dir,
            nframes=args.nframes,
            start_frame=args.start_frame,
            natoms_per_mol=args.natoms_per_mol,
            show_progress=not args.no_progress,
            save_every=None if args.save_every <= 0 else args.save_every,
            out_dir=args.out_dir,
            label=args.label,
            backend=args.backend,
            auto_build_fortran=args.auto_build_fortran,
        )
        out_paths = save_paircorr_results(
            results,
            out_dir=args.out_dir,
            label=args.label or gro_dir.name.replace(" ", "_"),
        )
        for path in out_paths:
            print(f"Wrote {path}")
        return 0

    if args.command == "build-fortran":
        built = build_paircorr_fortran(verbose=not args.quiet)
        print(f"Built {built}")
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
