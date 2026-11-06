#!/usr/bin/env python3
"""Analyze end-to-end nematic order in Tinker/AMOEBA 5CB ARC files.

Cpptraj constructs one C18-to-N01 vector for every 5CB residue after
autoimaging. Python normalizes those vectors, constructs the second-rank
ordering tensor in every frame, diagonalizes it, and averages its largest
eigenvalue over frames.

The external ARC tree is read only. All topology, cpptraj, numerical, VMD, and
plot products are written beneath the requested output directories.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATOMS_PER_MOLECULE = 38
START_LOCAL_ATOM = 19  # C18, terminal pentyl carbon (one based)
END_LOCAL_ATOM = 4  # N01, nitrile nitrogen (one based)

# Names follow the established 38-atom 5CB map. Only C18 and N01 are required
# by multivector, but naming the whole residue makes the generated topology and
# VMD view understandable.
ATOM_NAMES = (
    "C01", "C07", "C13", "N01", "C02", "C08", "C03", "C09", "C04", "C10",
    "C05", "C11", "C06", "C12", "C14", "C15", "C16", "C17", "C18", "H01",
    "H02", "H03", "H04", "H05", "H06", "H07", "H10", "H13", "H16", "H19",
    "H08", "H14", "H09", "H15", "H11", "H17", "H12", "H18",
)


@dataclass(frozen=True)
class Run:
    label: str
    temperature: float
    trajectory: Path


@dataclass(frozen=True)
class Atom:
    serial: int
    element: str
    xyz: tuple[float, float, float]
    bonds: tuple[int, ...]


def quote_cpptraj(path: Path) -> str:
    text = str(path)
    if '"' in text:
        raise ValueError(f"Double quotes are unsupported in cpptraj paths: {path}")
    return f'"{text}"'


def discover_runs(data_root: Path) -> list[Run]:
    runs: list[Run] = []
    for trajectory in sorted(data_root.glob("*/5cb-x1000.arc")):
        label = trajectory.parent.name
        match = re.search(r"-(\d{3})$", label)
        if match is None:
            raise ValueError(f"Cannot infer temperature from run name: {label}")
        runs.append(Run(label, float(match.group(1)), trajectory.resolve()))
    if not runs:
        raise FileNotFoundError(f"No */5cb-x1000.arc files below {data_root}")
    return sorted(runs, key=lambda run: run.temperature)


def read_first_tinker_frame(path: Path) -> tuple[str, np.ndarray, list[Atom]]:
    with path.open() as handle:
        header = handle.readline().split(maxsplit=1)
        if not header:
            raise ValueError(f"Empty Tinker ARC: {path}")
        natoms = int(header[0])
        title = header[1].strip() if len(header) > 1 else ""
        box_fields = handle.readline().split()
        if len(box_fields) < 6:
            raise ValueError(f"Expected six Tinker box fields in {path}")
        box = np.asarray([float(value) for value in box_fields[:6]], dtype=float)
        atoms: list[Atom] = []
        for _ in range(natoms):
            fields = handle.readline().split()
            if len(fields) < 6:
                raise ValueError(f"Malformed Tinker atom line in {path}")
            atoms.append(
                Atom(
                    serial=int(fields[0]),
                    element=fields[1],
                    xyz=(float(fields[2]), float(fields[3]), float(fields[4])),
                    bonds=tuple(int(value) for value in fields[6:]),
                )
            )
    if natoms % ATOMS_PER_MOLECULE:
        raise ValueError(f"{natoms} atoms is not divisible by {ATOMS_PER_MOLECULE}")
    if len(ATOM_NAMES) != ATOMS_PER_MOLECULE:
        raise RuntimeError("Internal 5CB atom-name map has the wrong length")
    return title, box, atoms


def pdb_atom_line(atom: Atom, atom_name: str, residue: int) -> str:
    x, y, z = atom.xyz
    element = atom.element[:2].title()
    return (
        f"ATOM  {atom.serial:5d} {atom_name:>4s} 5CB A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def write_cpptraj_topology(source_arc: Path, output_pdb: Path) -> tuple[int, np.ndarray]:
    title, box, atoms = read_first_tinker_frame(source_arc)
    nmolecules = len(atoms) // ATOMS_PER_MOLECULE
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    a, b, c, alpha, beta, gamma = box
    with output_pdb.open("w") as handle:
        handle.write(f"REMARK GENERATED FROM FIRST FRAME OF {source_arc}\n")
        handle.write(f"REMARK {title}\n")
        handle.write(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
        )
        for atom in atoms:
            local_index = (atom.serial - 1) % ATOMS_PER_MOLECULE
            residue = (atom.serial - 1) // ATOMS_PER_MOLECULE + 1
            handle.write(pdb_atom_line(atom, ATOM_NAMES[local_index], residue))
        for atom in atoms:
            if atom.bonds:
                handle.write(f"CONECT{atom.serial:5d}{''.join(f'{bond:5d}' for bond in atom.bonds)}\n")
        handle.write("END\n")
    return nmolecules, box


def run_cpptraj(cpptraj: str, input_path: Path, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [cpptraj, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def write_vector_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    raw_vectors: Path,
    nmolecules: int,
    start: int,
    stop: int,
    stride: int,
) -> None:
    lines = [
        f"parm {quote_cpptraj(topology)}",
        f"trajin {quote_cpptraj(trajectory)} {start} {stop} {stride}",
        "autoimage",
        (
            f"multivector e2e resrange 1-{nmolecules} "
            f"name1 C18 name2 N01 out {quote_cpptraj(raw_vectors)}"
        ),
        f"precision {quote_cpptraj(raw_vectors)} 13 6",
        "run",
        "quit",
        "",
    ]
    path.write_text("\n".join(lines))


def load_cpptraj_vectors(path: Path, nmolecules: int) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    frames = table[:, 0].astype(int)
    values = table[:, 1:]
    if values.shape[1] == 6 * nmolecules:
        # Cpptraj vector datasets are written as Vx,Vy,Vz,Ox,Oy,Oz: the
        # vector components followed by the origin, not two endpoints.
        records = values.reshape(values.shape[0], nmolecules, 6)
        vectors = records[:, :, 0:3]
    elif values.shape[1] == 3 * nmolecules:
        # Supported defensively in case a cpptraj build writes only components.
        vectors = values.reshape(values.shape[0], nmolecules, 3)
    else:
        raise ValueError(
            f"Expected {3 * nmolecules} or {6 * nmolecules} vector columns in "
            f"{path}, found {values.shape[1]}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite vector component in {path}")
    return frames, vectors


def frame_order(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vectors, axis=2)
    if np.any(lengths == 0.0):
        raise ValueError("Zero-length end-to-end vector")
    unit = vectors / lengths[:, :, None]
    nmolecules = unit.shape[1]
    q = 1.5 * np.einsum("fmi,fmj->fij", unit, unit) / nmolecules
    q -= 0.5 * np.eye(3)[None, :, :]
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    s_values = eigenvalues[:, -1]
    directors = eigenvectors[:, :, -1]
    return s_values, directors, lengths, q, eigenvalues


def analyze_one_run(
    run: Run,
    topology: str,
    out_root: str,
    cpptraj: str,
    nmolecules: int,
    start: int,
    stop: int,
    stride: int,
    reuse_vectors: bool,
) -> dict[str, float | int | str]:
    out_dir = Path(out_root) / run.label
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_vectors = out_dir / "end_to_end_vectors.dat"
    input_path = out_dir / "end_to_end_vectors.cpptraj.in"
    log_path = out_dir / "end_to_end_vectors.cpptraj.log"
    write_vector_input(
        input_path,
        Path(topology),
        run.trajectory,
        raw_vectors,
        nmolecules,
        start,
        stop,
        stride,
    )
    if not reuse_vectors or not raw_vectors.is_file():
        run_cpptraj(cpptraj, input_path, log_path)

    frames, vectors = load_cpptraj_vectors(raw_vectors, nmolecules)
    s_values, directors, lengths, q_tensors, eigenvalues = frame_order(vectors)
    ddof = 1 if s_values.size > 1 else 0
    std_s = float(s_values.std(ddof=ddof))
    np.save(out_dir / "S_by_frame.npy", s_values)
    np.save(out_dir / "director_by_frame.npy", directors)
    np.save(out_dir / "Q_by_frame.npy", q_tensors)
    np.save(out_dir / "Q_eigenvalues_by_frame.npy", eigenvalues)
    np.savetxt(
        out_dir / "S_by_frame.csv",
        np.column_stack((frames, s_values, directors)),
        delimiter=",",
        header="frame,S,n_x,n_y,n_z",
        comments="",
        fmt=("%d", "%.10f", "%.10f", "%.10f", "%.10f"),
    )
    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "frames": int(s_values.size),
        "molecules": nmolecules,
        "mean_S": float(s_values.mean()),
        "std_S": std_s,
        "sem_S": float(std_s / np.sqrt(s_values.size)),
        "mean_length_A": float(lengths.mean()),
        "std_length_A": float(lengths.std(ddof=1)),
        "min_length_A": float(lengths.min()),
        "max_length_A": float(lengths.max()),
    }


def parse_vector_pdb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            points.append(
                np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                )
            )
    if len(points) != 2:
        raise ValueError(f"Expected two vector points in {path}, found {len(points)}")
    return points[0], points[1]


def write_visualization(
    run: Run,
    topology: Path,
    out_root: Path,
    cpptraj: str,
    frame: int,
) -> None:
    out_dir = out_root / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    molecule_pdb = out_dir / "5cb_molecule_frame.pdb"
    vector_pdb = out_dir / "5cb_end_to_end_vector_cpptraj.pdb"
    vector_parm = out_dir / "5cb_end_to_end_vector_cpptraj.parm7"
    raw_vector = out_dir / "5cb_end_to_end_vector.dat"
    input_path = out_dir / "extract_and_vector.cpptraj.in"
    log_path = out_dir / "extract_and_vector.cpptraj.log"
    input_path.write_text(
        "\n".join(
            [
                f"parm {quote_cpptraj(topology)}",
                f"trajin {quote_cpptraj(run.trajectory)} {frame} {frame}",
                "autoimage",
                (
                    "vector e2e mask :1@C18 :1@N01 magnitude "
                    f"out {quote_cpptraj(raw_vector)}"
                ),
                "strip !:1",
                f"trajout {quote_cpptraj(molecule_pdb)} pdb conectmode all",
                "run",
                (
                    f"writedata {quote_cpptraj(vector_pdb)} e2e vectraj trajfmt pdb "
                    f"parmout {quote_cpptraj(vector_parm)}"
                ),
                "quit",
                "",
            ]
        )
    )
    run_cpptraj(cpptraj, input_path, log_path)
    start, end = parse_vector_pdb(vector_pdb)
    length = float(np.linalg.norm(end - start))
    start_text = " ".join(f"{value:.3f}" for value in start)
    end_text = " ".join(f"{value:.3f}" for value in end)
    (out_dir / "view_end_to_end_axis.vmd").write_text(
        "# Run: vmd -e view_end_to_end_axis.vmd\n"
        "set here [file dirname [file normalize [info script]]]\n"
        "mol new [file join $here 5cb_molecule_frame.pdb] type pdb waitfor all\n"
        "mol delrep 0 top\n"
        "mol representation Licorice 0.16 18.0 18.0\n"
        "mol color ColorID 2\n"
        "mol selection all\n"
        "mol material Opaque\n"
        "mol addrep top\n"
        "set molid [molinfo top]\n"
        "graphics $molid color orange\n"
        f"set p0 {{{start_text}}}\n"
        f"set p1 {{{end_text}}}\n"
        "set dv [vecsub $p1 $p0]\n"
        "set shaftend [vecadd $p0 [vecscale 0.86 $dv]]\n"
        "graphics $molid cylinder $p0 $shaftend radius 0.20 resolution 24 filled yes\n"
        "graphics $molid cone $shaftend $p1 radius 0.48 resolution 24\n"
        "color Display Background white\n"
        "display projection Orthographic\n"
        "display resetview\n"
        "axes location Off\n"
    )
    (out_dir / "README.md").write_text(
        "# AMOEBA 5CB end-to-end visualization\n\n"
        f"Source run: `{run.label}` ({run.temperature:.0f} K), frame {frame}.\n\n"
        "Cpptraj calculated the raw vector from local atom 19 (`C18`) to local "
        f"atom 4 (`N01`). Its length in this molecule is {length:.3f} A. The VMD "
        "helper displays the molecule in grey and the vector in orange.\n\n"
        "```bash\n"
        "vmd -e view_end_to_end_axis.vmd\n"
        "```\n"
    )


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = (
        "run", "temperature_K", "frames", "molecules", "mean_S", "std_S", "sem_S",
        "mean_length_A", "std_length_A", "min_length_A", "max_length_A",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    runs: list[Run], rows: list[dict[str, float | int | str]], out_data: Path, out_plot: Path
) -> None:
    run_lines = "\n".join(
        f"- `{run.label}`: {run.temperature:.0f} K, `{run.trajectory}`" for run in runs
    )
    result_lines = "\n".join(
        f"- {float(row['temperature_K']):.0f} K: `<S> = {float(row['mean_S']):.8f}`"
        for row in rows
    )
    (out_data / "README.md").write_text(
        "# AMOEBA 5CB end-to-end nematic order\n\n"
        "The external Tinker ARC trajectories were read only. Cpptraj 7.6.2 "
        "autoimaged every frame and constructed one raw vector per molecule:\n\n"
        "```text\n"
        "v_i = r_N01 - r_C18\n"
        "u_i = v_i / |v_i|\n"
        "C18 = local atom 19 (terminal pentyl carbon)\n"
        "N01 = local atom 4 (nitrile nitrogen)\n"
        "```\n\n"
        "For each frame containing N = 1000 molecules, Python calculated and "
        "diagonalized:\n\n"
        "```text\n"
        "Q = (1/N) sum_i [ (3/2) u_i u_i - (1/2) I ]\n"
        "S_frame = largest eigenvalue of Q\n"
        "<S> = mean over the 1000 frame values\n"
        "```\n\n"
        "The plotted error bars are the naive standard error over frames and "
        "are not corrected for temporal autocorrelation. Each run directory "
        "retains the cpptraj input/log and raw vector table, plus "
        "per-frame Q tensors, eigenvalues, directors, and S values.\n\n"
        "## Sources\n\n"
        f"{run_lines}\n\n"
        "## Results\n\n"
        f"{result_lines}\n\n"
        "## Main outputs\n\n"
        f"- Summary: `{out_data / 'summary.csv'}`\n"
        f"- Plot: `{out_plot}`\n"
        f"- Eigenvalue time-series panel: `{out_plot.parent / 'eigenvalues_timeseries_panel.png'}`\n"
        f"- Visualization: `{out_data / 'visualization/view_end_to_end_axis.vmd'}`\n"
    )


def make_plot(rows: list[dict[str, float | int | str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    means = np.asarray([float(row["mean_S"]) for row in rows])
    sems = np.asarray([float(row["sem_S"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.errorbar(
        temperatures,
        means,
        yerr=sems,
        color="#d95f02",
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
        label="C18 -> N01 end-to-end axis",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean nematic order, <S>")
    ax.set_xticks(temperatures)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", "--data-root", dest="data_root", type=Path, required=True)
    parser.add_argument("--out-data", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visualization-frame", type=int, default=1)
    parser.add_argument("--reuse-vectors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.data_root.resolve())
    out_data = args.out_data.resolve()
    out_data.mkdir(parents=True, exist_ok=True)
    cpptraj = shutil.which(args.cpptraj)
    if cpptraj is None:
        raise SystemExit(f"cpptraj executable not found: {args.cpptraj}")

    topology = out_data / "topology" / "5cb-x1000-first-frame.pdb"
    nmolecules, _ = write_cpptraj_topology(runs[0].trajectory, topology)
    if nmolecules != 1000:
        raise ValueError(f"Expected 1000 5CB molecules, found {nmolecules}")
    write_visualization(
        runs[0], topology, out_data, cpptraj, frame=args.visualization_frame
    )

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_one_run,
                run,
                str(topology),
                str(out_data),
                cpptraj,
                nmolecules,
                args.start,
                args.stop,
                args.stride,
                args.reuse_vectors,
            ): run
            for run in runs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['run']} ({float(row['temperature_K']):.0f} K): "
                f"{row['frames']} frames, <S>={float(row['mean_S']):.6f}",
                flush=True,
            )

    rows.sort(key=lambda row: float(row["temperature_K"]))
    write_summary(rows, out_data / "summary.csv")
    out_plot = args.out_plot.resolve()
    make_plot(rows, out_plot)
    write_readme(runs, rows, out_data, out_plot)
    print(f"Wrote {out_data / 'summary.csv'}")
    print(f"Wrote {args.out_plot.resolve()}")


if __name__ == "__main__":
    main()
