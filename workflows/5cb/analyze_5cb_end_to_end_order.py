#!/usr/bin/env python3
"""Calculate 5CB end-to-end nematic order and make a vector visualization.

The 5CB axis is defined from local atom 19 (C18, terminal pentyl carbon) to
local atom 4 (N01, nitrile nitrogen).  Source trajectories are read only;
all generated files are written beneath the requested output directories.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nematics.datasets import get_dataset  # noqa: E402
from nematics.gro import read_gro_frame  # noqa: E402
from nematics.paf import compute_global_paf, make_end_to_end_axes  # noqa: E402


GAFF_5CB = get_dataset("gaff-5cb")
RUNS = tuple((run.id, run.temperature_k) for run in GAFF_5CB.runs)
NATOMS_PER_MOLECULE = 38
START_ATOM = 18  # zero-based: local atom 19, C18
END_ATOM = 3  # zero-based: local atom 4, N01


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)\.gro", path.name)
    if match is None:
        raise ValueError(f"Not a numbered GRO frame: {path}")
    return int(match.group(1))


def analyze_run(run_dir: str, nframes: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame S and director arrays for one trajectory directory."""
    directory = Path(run_dir)
    frames = sorted(directory.glob("frame*.gro"), key=frame_number)
    if nframes is not None:
        frames = frames[:nframes]
    if not frames:
        raise FileNotFoundError(f"No frame*.gro files in {directory}")

    s_values = np.empty(len(frames), dtype=float)
    directors = np.empty((len(frames), 3), dtype=float)
    for index, path in enumerate(frames):
        frame = read_gro_frame(path, natoms_per_mol=NATOMS_PER_MOLECULE)
        axes = make_end_to_end_axes(
            frame.positions,
            start_atom=START_ATOM,
            end_atom=END_ATOM,
            box=frame.box,
        )
        eigenvalues, eigenvectors = compute_global_paf(axes)
        s_values[index] = eigenvalues[0]
        directors[index] = eigenvectors[:, 0]
    return s_values, directors


def read_first_molecule_metadata(path: Path) -> tuple[list[str], list[str]]:
    atom_names: list[str] = []
    elements: list[str] = []
    with path.open() as handle:
        handle.readline()
        natoms = int(handle.readline().strip())
        if natoms < NATOMS_PER_MOLECULE:
            raise ValueError(f"Too few atoms in {path}")
        for _ in range(NATOMS_PER_MOLECULE):
            line = handle.readline()
            name = line[10:15].strip()
            atom_names.append(name)
            elements.append("N" if name.startswith("N") else ("H" if name.startswith("H") else "C"))
    return atom_names, elements


def pdb_atom_line(
    record: str,
    serial: int,
    name: str,
    residue: str,
    residue_number: int,
    xyz: np.ndarray,
    element: str,
) -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {name:>4s} {residue:>3s} A{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def write_visualization(source_frame: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_gro_frame(source_frame, natoms_per_mol=NATOMS_PER_MOLECULE)
    atom_names, elements = read_first_molecule_metadata(source_frame)

    molecule = frame.positions[0].copy()
    anchor = molecule[START_ATOM].copy()
    molecule = anchor + (molecule - anchor) - frame.box * np.rint((molecule - anchor) / frame.box)
    molecule_angstrom = 10.0 * molecule
    # Copy because the molecule array is translated in place below.
    start = molecule_angstrom[START_ATOM].copy()
    endpoint_displacement = frame.positions[0, END_ATOM] - frame.positions[0, START_ATOM]
    endpoint_displacement -= frame.box * np.rint(endpoint_displacement / frame.box)
    end = start + 10.0 * endpoint_displacement

    # Put the molecule near the origin for convenient viewing.
    center = molecule_angstrom.mean(axis=0)
    molecule_angstrom -= center
    start -= center
    end -= center

    pdb_path = output_dir / "5cb_end_to_end_axis.pdb"
    vector_points = np.linspace(start, end, 13)
    with pdb_path.open("w") as handle:
        handle.write("REMARK 5CB LONG AXIS: LOCAL ATOM 19 C18 TO LOCAL ATOM 4 N01\n")
        handle.write("REMARK VECTOR DUMMY ATOMS HAVE RESNAME VEC AND LIE ON THE UNNORMALIZED AXIS\n")
        for serial, (name, element, xyz) in enumerate(
            zip(atom_names, elements, molecule_angstrom, strict=True), start=1
        ):
            handle.write(pdb_atom_line("ATOM", serial, name, "5CB", 1, xyz, element))
        first_vector_serial = NATOMS_PER_MOLECULE + 1
        for offset, xyz in enumerate(vector_points):
            serial = first_vector_serial + offset
            handle.write(pdb_atom_line("HETATM", serial, f"V{offset + 1:02d}", "VEC", 2, xyz, "X"))
        for serial in range(first_vector_serial, first_vector_serial + len(vector_points) - 1):
            handle.write(f"CONECT{serial:5d}{serial + 1:5d}\n")
        handle.write("END\n")

    start_text = " ".join(f"{value:.3f}" for value in start)
    end_text = " ".join(f"{value:.3f}" for value in end)
    vmd_path = output_dir / "view_5cb_end_to_end_axis.vmd"
    vmd_path.write_text(
        "# Run: vmd -e view_5cb_end_to_end_axis.vmd\n"
        "set here [file dirname [file normalize [info script]]]\n"
        "mol new [file join $here 5cb_end_to_end_axis.pdb] type pdb waitfor all\n"
        "mol delrep 0 top\n"
        "mol representation Licorice 0.16 18.0 18.0\n"
        "mol color ColorID 2\n"
        "mol selection {resname 5CB}\n"
        "mol material Opaque\n"
        "mol addrep top\n"
        "mol representation VDW 0.22 18.0\n"
        "mol color ColorID 3\n"
        "mol selection {resname VEC}\n"
        "mol addrep top\n"
        "set molid [molinfo top]\n"
        "graphics $molid color orange\n"
        f"set p0 {{{start_text}}}\n"
        f"set p1 {{{end_text}}}\n"
        "set dv [vecsub $p1 $p0]\n"
        "set shaftend [vecadd $p0 [vecscale 0.86 $dv]]\n"
        "graphics $molid cylinder $p0 $shaftend radius 0.20 resolution 24 filled yes\n"
        "graphics $molid cone $shaftend $p1 radius 0.48 resolution 24\n"
        "display projection Orthographic\n"
        "display resetview\n"
        "axes location Off\n"
        "color Display Background white\n"
    )

    length = float(np.linalg.norm(end - start))
    (output_dir / "README.md").write_text(
        "# 5CB end-to-end long axis\n\n"
        "The orange vector points from local atom 19 (`C18`, the terminal pentyl "
        "carbon) to local atom 4 (`N01`, the nitrile nitrogen). The vector in this "
        f"example is {length:.3f} A long before normalization.\n\n"
        "Load the prepared view with:\n\n"
        "```bash\n"
        "vmd -e view_5cb_end_to_end_axis.vmd\n"
        "```\n"
    )


def write_summary(results: dict[str, tuple[float, np.ndarray, np.ndarray]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "temperature_K", "nframes", "mean_S", "std_S", "sem_S"))
        for run, temperature in RUNS:
            _, values, _ = results[run]
            writer.writerow(
                (
                    run,
                    f"{temperature:.1f}",
                    values.size,
                    f"{values.mean():.10f}",
                    f"{values.std(ddof=1):.10f}",
                    f"{values.std(ddof=1) / np.sqrt(values.size):.10f}",
                )
            )


def make_plot(results: dict[str, tuple[float, np.ndarray, np.ndarray]], output: Path) -> None:
    import matplotlib.pyplot as plt

    temperatures = np.array([temperature for _, temperature in RUNS])
    means = np.array([results[run][1].mean() for run, _ in RUNS])
    sems = np.array(
        [results[run][1].std(ddof=1) / np.sqrt(results[run][1].size) for run, _ in RUNS]
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.errorbar(
        temperatures,
        means,
        yerr=sems,
        color="#d95f02",
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        label="C18 -> N01 end-to-end axis",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean nematic order, <S>")
    ax.set_xticks(temperatures)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        "--data-root",
        dest="data_root",
        type=Path,
        default=GAFF_5CB.raw_path(),
    )
    parser.add_argument("--out-data", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_data.mkdir(parents=True, exist_ok=True)

    write_visualization(args.data_root / "gro" / "frame0.gro", args.out_data / "visualization")

    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_run, str(args.data_root / run), args.nframes): (run, temperature)
            for run, temperature in RUNS
        }
        for future in as_completed(futures):
            run, temperature = futures[future]
            values, directors = future.result()
            results[run] = (temperature, values, directors)
            np.save(args.out_data / f"{run}_end_to_end_S.npy", values)
            np.save(args.out_data / f"{run}_end_to_end_director.npy", directors)
            print(
                f"{run} ({temperature:.0f} K): {values.size} frames, "
                f"<S>={values.mean():.6f}",
                flush=True,
            )

    write_summary(results, args.out_data / "summary.csv")
    make_plot(results, args.out_plot)
    print(f"Wrote {args.out_data / 'summary.csv'}")
    print(f"Wrote {args.out_plot}")


if __name__ == "__main__":
    main()
