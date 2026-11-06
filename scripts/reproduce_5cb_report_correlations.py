#!/usr/bin/env python3
"""Reproduce the four 5CB correlation-function figures in try_to_reproduce.pdf."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from nematics.fortran_backend import load_paircorr_fortran
from nematics.gro import read_gro_frame
from nematics.paf import compute_global_paf, make_molecular_axes
from nematics.paircorr import CLOSE_IMAGES


ARRAY_SPECS = {
    "global_par_1": (200, 4),
    "global_par_2": (200, 4),
    "global_par_3": (200, 4),
    "local_par_1": (400, 4),
    "local_par_2": (400, 4),
    "local_par_3": (400, 4),
    "global_per": (200, 4),
    "local_per": (200, 4),
    "cn_rad": (200, 4),
}

REPORT_RUNS = (
    ("gro4", 386),
    ("gro3", 376),
    ("gro2", 366),
    ("gro", 356),
)


def centers(minimum: float, maximum: float, size: int) -> np.ndarray:
    width = (maximum - minimum) / size
    return minimum + width * (np.arange(size, dtype=float) + 0.5)


def fresh_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float64, order="F")
        for name, shape in ARRAY_SPECS.items()
    }


def save_state(
    path: Path,
    arrays: dict[str, np.ndarray],
    nsnaps: int,
    label: str,
    temperature: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        nsnaps=np.asarray(nsnaps),
        label=np.asarray(label),
        temperature_K=np.asarray(temperature),
    )


def load_state(path: Path) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path) as saved:
        arrays = {
            name: np.asfortranarray(saved[name], dtype=np.float64)
            for name in ARRAY_SPECS
        }
        nsnaps = int(saved["nsnaps"])
    return arrays, nsnaps


def save_tables(out_dir: Path, arrays: dict[str, np.ndarray], nsnaps: int) -> None:
    for name, array in arrays.items():
        if name.startswith("local_par"):
            x = centers(-5.0, 5.0, array.shape[0])
        else:
            x = centers(0.0, 5.0, array.shape[0])
        normalized = array / float(nsnaps)
        np.savetxt(
            out_dir / f"{name}.csv",
            np.column_stack((x, normalized)),
            delimiter=",",
            header="coordinate_nm,raw_pairs_per_frame,g0,g1,g2",
            comments="",
            fmt="%.10g",
        )


def analyze(args: argparse.Namespace) -> int:
    gro_dir = Path(args.gro_dir).resolve()
    out_dir = Path(args.out_dir).resolve() / args.label
    state_path = out_dir / "correlations.npz"
    if args.resume and state_path.exists():
        arrays, nsnaps = load_state(state_path)
    else:
        arrays = fresh_arrays()
        nsnaps = 0

    if nsnaps > args.nframes:
        raise SystemExit(f"Checkpoint contains {nsnaps} frames, requested {args.nframes}")

    fort = load_paircorr_fortran(auto_build=False)
    if fort is None or not hasattr(fort.paircorr_fort_mod, "report_paircorr_update"):
        raise SystemExit("Updated Fortran backend is unavailable; run 'nematics build-fortran'")

    close_images = np.asfortranarray(CLOSE_IMAGES, dtype=np.float64)
    update = fort.paircorr_fort_mod.report_paircorr_update
    for frame_index in range(nsnaps, args.nframes):
        frame = read_gro_frame(
            gro_dir / f"frame{frame_index}.gro",
            natoms_per_mol=args.natoms_per_mol,
        )
        axes = make_molecular_axes(frame.positions)
        _, director_axes = compute_global_paf(axes.ddir)
        global_director = np.asfortranarray(director_axes[:, 0], dtype=np.float64)

        update(
            np.asfortranarray(axes.molcm, dtype=np.float64),
            np.asfortranarray(axes.molzax, dtype=np.float64),
            np.asfortranarray(axes.dpos, dtype=np.float64),
            np.asfortranarray(axes.ddir, dtype=np.float64),
            np.asfortranarray(frame.box, dtype=np.float64),
            global_director,
            close_images,
            float(args.slab_half_thickness),
            arrays["global_par_1"],
            arrays["global_par_2"],
            arrays["global_par_3"],
            arrays["local_par_1"],
            arrays["local_par_2"],
            arrays["local_par_3"],
            arrays["global_per"],
            arrays["local_per"],
            arrays["cn_rad"],
        )
        nsnaps = frame_index + 1
        if nsnaps % args.save_every == 0 or nsnaps == args.nframes:
            save_state(state_path, arrays, nsnaps, args.label, args.temperature)
            print(f"{args.label}: saved {nsnaps}/{args.nframes} frames", flush=True)

    save_tables(out_dir, arrays, nsnaps)
    summary = (
        f"label: {args.label}\n"
        f"temperature_K: {args.temperature} (inferred)\n"
        f"gro_directory: {gro_dir}\n"
        f"frames: {nsnaps}\n"
        f"molecules: 1000\n"
        f"atoms_per_molecule: {args.natoms_per_mol}\n"
        f"slab_half_thickness_nm: {args.slab_half_thickness}\n"
        "director: largest PAF eigenvector from CN directions (atoms 3-4)\n"
        "molecular_axis: unit vector from local atom 10 to local atom 9\n"
        "channels: g0, g1=<u_i.u_j>, g2=<P2(u_i.u_j)>\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary, end="")
    return 0


def load_run(input_root: Path, label: str) -> dict[str, np.ndarray | int]:
    path = input_root / label / "correlations.npz"
    with np.load(path) as saved:
        nsnaps = int(saved["nsnaps"])
        result: dict[str, np.ndarray | int] = {"nsnaps": nsnaps}
        for name in ARRAY_SPECS:
            result[name] = np.asarray(saved[name], dtype=float) / nsnaps
    return result


def style_axis(axis: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.axhline(0.0, color="0.65", linewidth=0.7)
    axis.grid(axis="x", color="0.72", linestyle=":", linewidth=0.7)
    axis.tick_params(direction="in", top=True, right=True)


def plot_parallel(
    runs: dict[str, dict[str, np.ndarray | int]],
    out_path: Path,
    local: bool,
) -> None:
    colors = {1: "#a000ff", 2: "#00a878", 3: "#23a6d5"}
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 7.2), sharex=True, sharey="row")
    for column, (label, temperature) in enumerate(REPORT_RUNS):
        run = runs[label]
        for radius in (1, 2, 3):
            key = f"{'local' if local else 'global'}_par_{radius}"
            array = np.asarray(run[key])
            if local:
                x = centers(-5.0, 5.0, array.shape[0])
                values = array[:, 1:4]
            else:
                positive_x = centers(0.0, 5.0, array.shape[0])
                x = np.concatenate((-positive_x[::-1], positive_x))
                values = np.concatenate((array[::-1, 1:4], array[:, 1:4]), axis=0)
            for row in range(3):
                axes[row, column].plot(
                    x,
                    values[:, row],
                    color=colors[radius],
                    linewidth=1.15,
                    label=fr"$R_0={radius}$ nm",
                )
        axes[0, column].set_title(f"{temperature} K")

    ylims = ((0.76, 1.10), (-0.15, 0.13), (-0.03, 0.62))
    for row in range(3):
        for column in range(4):
            style_axis(axes[row, column], (-5.0, 5.0), ylims[row])
        axes[row, 0].set_ylabel(fr"$g_{row}^{{\parallel}}$")
    for axis in axes[-1, :]:
        axis.set_xlabel("z (nm)")
    axes[2, 0].legend(frameon=False, fontsize=9, loc="upper left")
    title = "Pair correlations along the molecular z-axis" if local else "Pair correlations along the nematic director"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=0.0, w_pad=0.0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_perpendicular(runs: dict[str, dict[str, np.ndarray | int]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 7.2), sharex=True, sharey="row")
    x = centers(0.0, 5.0, 200)
    for column, (label, temperature) in enumerate(REPORT_RUNS):
        run = runs[label]
        global_per = np.asarray(run["global_per"])
        local_per = np.asarray(run["local_per"])
        for row in range(3):
            axes[row, column].plot(x, global_per[:, row + 1], color="#a000ff", linewidth=1.15, label="perpendicular n")
            axes[row, column].plot(x, local_per[:, row + 1], color="#00a878", linewidth=1.15, label="perpendicular mol-z")
        axes[0, column].set_title(f"{temperature} K")

    ylims = ((0.0, 1.65), (-0.16, 0.25), (-0.03, 0.82))
    for row in range(3):
        for column in range(4):
            style_axis(axes[row, column], (0.0, 5.0), ylims[row])
        axes[row, 0].set_ylabel(fr"$g_{row}^{{\perp}}$")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$r_{\perp}$ (nm)")
    axes[2, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Pair correlations perpendicular to the aligned axis", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=0.0, w_pad=0.0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cn_radial(runs: dict[str, dict[str, np.ndarray | int]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 7.2), sharex=True, sharey="row")
    x = centers(0.0, 5.0, 200)
    for column, (label, temperature) in enumerate(REPORT_RUNS):
        cn = np.asarray(runs[label]["cn_rad"])
        for row in range(3):
            axes[row, column].plot(x, cn[:, row + 1], color="#a000c8", linewidth=1.3)
        axes[1, column].fill_between(x, 0.0, cn[:, 2], color="#a000c8", alpha=0.18)
        axes[0, column].set_title(f"{temperature} K")

    ylims = ((0.0, 1.22), (-0.27, 0.13), (-0.03, 0.67))
    for row in range(3):
        for column in range(4):
            style_axis(axes[row, column], (0.0, 5.0), ylims[row])
        axes[row, 0].set_ylabel(fr"$g_{row}^{{\mathrm{{rad}}}}$")
    for axis in axes[-1, :]:
        axis.set_xlabel("r (nm)")
    fig.suptitle("CN dipole-dipole radial correlations (inter-dipole distance)", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=0.0, w_pad=0.0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {label: load_run(input_root, label) for label, _ in REPORT_RUNS}
    plot_parallel(runs, out_dir / "5cb_along_nematic_director.png", local=False)
    plot_parallel(runs, out_dir / "5cb_along_molecular_z.png", local=True)
    plot_perpendicular(runs, out_dir / "5cb_perpendicular_global_vs_molecular.png")
    plot_cn_radial(runs, out_dir / "5cb_cn_dipole_radial.png")
    for path in sorted(out_dir.glob("*.png")):
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--gro-dir", required=True)
    analysis.add_argument("--label", required=True)
    analysis.add_argument("--temperature", type=int, required=True)
    analysis.add_argument("--out-dir", required=True)
    analysis.add_argument("--nframes", type=int, default=1000)
    analysis.add_argument("--natoms-per-mol", type=int, default=38)
    analysis.add_argument("--slab-half-thickness", type=float, default=0.2)
    analysis.add_argument("--save-every", type=int, default=25)
    analysis.add_argument("--resume", action="store_true")

    plotting = subparsers.add_parser("plot")
    plotting.add_argument("--input-root", required=True)
    plotting.add_argument("--out-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "analyze":
        return analyze(args)
    return plot(args)


if __name__ == "__main__":
    raise SystemExit(main())

