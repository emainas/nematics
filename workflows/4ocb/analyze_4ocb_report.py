#!/usr/bin/env python3
"""AMOEBA-only 4OCB processing used to build the 4OCB results report.

The command has four independently reusable stages: ``order``, ``rdf``,
``paircorr``, and ``dihedral``.  All source ARC files are read only; cpptraj
inputs, logs, raw tables, reductions, and figures are written in this repo.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "workflows" / "5cb"))

from analyze_5cb_g1_director_paircorr import (  # noqa: E402
    read_g1,
    write_table as write_g1_table,
)
from analyze_amoeba_5cb_director_paircorr import (  # noqa: E402
    RADII_NM,
    candidate_pairs,
    director_basis,
    director_from_axes,
    histogram_pair_values,
    load_cpptraj_axes,
    load_cpptraj_centers,
    read_result,
    write_center_input,
    write_table as write_g2_table,
)
from analyze_amoeba_5cb_inertia_order import load_principal_axes  # noqa: E402
from nematics.datasets import get_dataset  # noqa: E402


DATASET = get_dataset("amoeba-4ocb")
NMOLECULES = DATASET.molecules
ATOMS_PER_MOLECULE = DATASET.atoms_per_molecule
R_KCAL = 0.00198720425864083

# One-based local atom map inferred from the first ARC frame and its bonds:
# nitrile: 5-14-2; biphenyl: 1-8; butoxy: 11-28-15-16-17-18.
END_TO_END_NAMES = ("C17", "N01")  # terminal butyl carbon -> nitrile N
CORE_NAMES = ("C10", "C04")  # butoxy-side ipso -> nitrile-side ipso
ETHER_DIHEDRAL = (16, 15, 28, 11)  # C-C-O-C_ar; central bond 15-28
RING_JUNCTION_DIHEDRAL = (10, 11, 28, 15)  # C_ortho-C_ipso-O-C; bond 11-28

ATOM_NAMES = (
    "C01", "N01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
    "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
    "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09",
    "O01", "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17",
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


def discover_runs(root: Path) -> list[Run]:
    return [
        Run(run.id, run.temperature_k, trajectory.resolve())
        for run, trajectory in DATASET.run_trajectories(root)
    ]


def run_cpptraj(executable: str, input_path: Path, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [executable, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def read_first_frame(path: Path) -> tuple[str, np.ndarray, list[Atom]]:
    with path.open() as handle:
        header = handle.readline().split(maxsplit=1)
        natoms = int(header[0])
        title = header[1].strip() if len(header) > 1 else ""
        box = np.asarray([float(value) for value in handle.readline().split()[:6]])
        atoms = []
        for _ in range(natoms):
            fields = handle.readline().split()
            atoms.append(
                Atom(
                    serial=int(fields[0]),
                    element=fields[1],
                    xyz=tuple(float(value) for value in fields[2:5]),
                    bonds=tuple(int(value) for value in fields[6:]),
                )
            )
    if natoms % ATOMS_PER_MOLECULE:
        raise ValueError(f"{natoms} atoms is not divisible by {ATOMS_PER_MOLECULE}")
    return title, box, atoms


def ensure_topology(runs: list[Run], topology: Path) -> Path:
    if topology.is_file():
        return topology
    title, box, atoms = read_first_frame(runs[0].trajectory)
    topology.parent.mkdir(parents=True, exist_ok=True)
    a, b, c, alpha, beta, gamma = box
    with topology.open("w") as handle:
        handle.write(f"REMARK GENERATED FROM FIRST FRAME OF {runs[0].trajectory}\n")
        handle.write(f"REMARK {title}\n")
        handle.write(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
        )
        for atom in atoms:
            local = (atom.serial - 1) % ATOMS_PER_MOLECULE
            residue = (atom.serial - 1) // ATOMS_PER_MOLECULE + 1
            x, y, z = atom.xyz
            handle.write(
                f"ATOM  {atom.serial:5d} {ATOM_NAMES[local]:>4s} 4OC A{residue:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          "
                f"{atom.element[:2].title():>2s}\n"
            )
        for atom in atoms:
            if atom.bonds:
                handle.write(
                    f"CONECT{atom.serial:5d}{''.join(f'{bond:5d}' for bond in atom.bonds)}\n"
                )
        handle.write("END\n")
    return topology


def frame_order(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = 1.5 * np.einsum("fmi,fmj->fij", axes, axes) / axes.shape[1]
    q -= 0.5 * np.eye(3)[None, :, :]
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    return q, eigenvalues, eigenvectors[:, :, -1]


def save_axis_order(output: Path, frames: np.ndarray, axes: np.ndarray) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    q, eigenvalues, directors = frame_order(axes)
    s_values = eigenvalues[:, -1]
    np.save(output / "axes.npy", axes)
    np.save(output / "Q_by_frame.npy", q)
    np.save(output / "eigenvalues_by_frame.npy", eigenvalues)
    np.save(output / "director_by_frame.npy", directors)
    np.save(output / "S_by_frame.npy", s_values)
    np.savetxt(
        output / "S_by_frame.csv",
        np.column_stack((frames, s_values)),
        delimiter=",",
        header="frame,S",
        comments="",
        fmt=("%d", "%.10f"),
    )
    std = float(s_values.std(ddof=1)) if len(s_values) > 1 else 0.0
    return {
        "mean": float(s_values.mean()),
        "std": std,
        "sem": std / np.sqrt(len(s_values)),
    }


def write_order_input(
    path: Path,
    topology: Path,
    run: Run,
    output: Path,
    start: int,
    stop: int,
    stride: int,
) -> None:
    end_path = output / "end_to_end_vectors.dat"
    core_path = output / "core_vectors.dat"
    inertia_path = output / "inertia_vectors.dat"
    lines = [
        f"parm {quote_cpptraj(topology)}",
        f"trajin {quote_cpptraj(run.trajectory)} {start} {stop} {stride}",
        "autoimage",
        (
            f"multivector end_to_end resrange 1-{NMOLECULES} "
            f"name1 {END_TO_END_NAMES[0]} name2 {END_TO_END_NAMES[1]} "
            f"out {quote_cpptraj(end_path)}"
        ),
        (
            f"multivector core resrange 1-{NMOLECULES} "
            f"name1 {CORE_NAMES[0]} name2 {CORE_NAMES[1]} out {quote_cpptraj(core_path)}"
        ),
    ]
    lines.extend(
        f"vector inertia_z_{residue:04d} principal z :{residue} out {quote_cpptraj(inertia_path)}"
        for residue in range(1, NMOLECULES + 1)
    )
    lines.extend(
        [
            f"precision {quote_cpptraj(end_path)} 14 8",
            f"precision {quote_cpptraj(core_path)} 14 8",
            f"precision {quote_cpptraj(inertia_path)} 14 8",
            "run",
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def process_order_run(
    run: Run,
    topology: str,
    output_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    force: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / run.label
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "order_axes.cpptraj.in"
    log_path = output / "order_axes.cpptraj.log"
    write_order_input(input_path, Path(topology), run, output, start, stop, stride)
    raw_paths = {
        "end_to_end": output / "end_to_end_vectors.dat",
        "core": output / "core_vectors.dat",
        "inertia": output / "inertia_vectors.dat",
    }
    reused = all(path.is_file() for path in raw_paths.values()) and not force
    if not reused:
        run_cpptraj(cpptraj, input_path, log_path)

    end_frames, end_axes = load_cpptraj_axes(raw_paths["end_to_end"])
    core_frames, core_axes = load_cpptraj_axes(raw_paths["core"])
    inertia_frames, inertia_axes, _ = load_principal_axes(
        raw_paths["inertia"], NMOLECULES
    )
    if not np.array_equal(end_frames, core_frames) or not np.array_equal(
        end_frames, inertia_frames
    ):
        raise ValueError(f"Axis frame mismatch for {run.label}")

    results = {
        "end_to_end": save_axis_order(output / "end_to_end", end_frames, end_axes),
        "core": save_axis_order(output / "core", core_frames, core_axes),
        "inertia": save_axis_order(output / "inertia", inertia_frames, inertia_axes),
    }
    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "source_arc": str(run.trajectory),
        "frames": len(end_frames),
        "molecules": NMOLECULES,
        "mean_S_end_to_end": results["end_to_end"]["mean"],
        "sem_S_end_to_end": results["end_to_end"]["sem"],
        "mean_S_core": results["core"]["mean"],
        "sem_S_core": results["core"]["sem"],
        "mean_S_inertia": results["inertia"]["mean"],
        "sem_S_inertia": results["inertia"]["sem"],
        "cpptraj_reused": str(reused),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_order(rows: list[dict[str, object]], path: Path) -> None:
    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    styles = (
        ("end_to_end", "terminal C → N", "#d95f02", "o"),
        ("inertia", "inertia, smallest moment", "#1f9e89", "s"),
        ("core", "biphenyl core", "#5e3c99", "^"),
    )
    for key, label, color, marker in styles:
        means = np.asarray([float(row[f"mean_S_{key}"]) for row in rows])
        sems = np.asarray([float(row[f"sem_S_{key}"]) for row in rows])
        ax.errorbar(
            temperatures,
            means,
            yerr=sems,
            color=color,
            marker=marker,
            linewidth=1.8,
            capsize=2.5,
            label=label,
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean nematic order, <S>")
    ax.set_xticks(temperatures)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def resolve_cpptraj(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise SystemExit(
            f"cpptraj executable not found: {value}; load an available AmberTools "
            "module or pass --cpptraj /path/to/cpptraj"
        )
    return resolved


def command_order(args: argparse.Namespace) -> None:
    runs = discover_runs(args.raw_root.resolve())
    topology = ensure_topology(runs, args.topology.resolve())
    cpptraj = resolve_cpptraj(args.cpptraj)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_order_run,
                run,
                str(topology),
                str(args.out_data.resolve()),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.force,
            ): run.label
            for run in runs
        }
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"Finished order: {futures[future]}", flush=True)
    rows.sort(key=lambda row: float(row["temperature_K"]))
    write_csv(args.out_data.resolve() / "summary.csv", rows)
    plot_order(rows, args.out_plot.resolve())


def write_rdf_input(
    path: Path,
    topology: Path,
    run: Run,
    rdf: Path,
    raw: Path,
    start: int,
    stop: int,
    stride: int,
    spacing_a: float,
    maximum_a: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"parm {quote_cpptraj(topology)}",
                f"trajin {quote_cpptraj(run.trajectory)} {start} {stop} {stride}",
                "autoimage",
                (
                    f"radial out {quote_cpptraj(rdf)} {spacing_a:g} {maximum_a:g} "
                    f":1-{NMOLECULES} :1-{NMOLECULES} bymol1 bymol2 mass volume "
                    f"rawrdf {quote_cpptraj(raw)}"
                ),
                f"precision {quote_cpptraj(rdf)} 14 8",
                f"precision {quote_cpptraj(raw)} 18 6",
                "run",
                "quit",
                "",
            ]
        )
    )


def process_rdf_run(
    run: Run,
    topology: str,
    output_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    spacing_a: float,
    maximum_a: float,
    force: bool,
) -> dict[str, object]:
    output = Path(output_root) / run.label
    output.mkdir(parents=True, exist_ok=True)
    rdf = output / "com_com_rdf_cpptraj.dat"
    raw = output / "com_com_rdf_raw_counts.dat"
    input_path = output / "com_com_rdf.cpptraj.in"
    log = output / "com_com_rdf.cpptraj.log"
    write_rdf_input(
        input_path, Path(topology), run, rdf, raw, start, stop, stride, spacing_a, maximum_a
    )
    reused = rdf.is_file() and raw.is_file() and not force
    if not reused:
        run_cpptraj(cpptraj, input_path, log)
    values = np.loadtxt(rdf, comments="#", ndmin=2)
    counts = np.loadtxt(raw, comments="#", ndmin=2)
    if values.shape != counts.shape or not np.allclose(values[:, 0], counts[:, 0]):
        raise ValueError(f"RDF grid mismatch for {run.label}")
    csv_path = output / "com_com_rdf.csv"
    np.savetxt(
        csv_path,
        np.column_stack((values[:, 0] / 10.0, values[:, 1], counts[:, 1])),
        delimiter=",",
        header="r_nm,g_com_com,raw_pair_count",
        comments="",
        fmt=("%.6f", "%.10f", "%.0f"),
    )
    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "source_arc": str(run.trajectory),
        "frames": (stop - start) // stride + 1,
        "molecules": NMOLECULES,
        "csv": str(csv_path),
        "cpptraj_reused": str(reused),
    }


def plot_rdf(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("turbo", len(rows))
    maximum = 0.0
    curves = []
    for row in rows:
        table = np.loadtxt(str(row["csv"]), delimiter=",", skiprows=1)
        curves.append(table)
        maximum = max(maximum, float(table[:, 1].max()))
    for index, (ax, row, table) in enumerate(zip(axes.ravel(), rows, curves, strict=False)):
        ax.plot(table[:, 0], table[:, 1], color=cmap(index), linewidth=1.5)
        ax.axhline(1.0, color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{float(row['temperature_K']):.0f} K")
        ax.set_ylim(0.0, 1.08 * maximum)
        ax.grid(alpha=0.18)
    axes.ravel()[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("g_COM-COM(r)")
    for ax in axes[-1, :2]:
        ax.set_xlabel("COM-COM distance r (nm)")
    fig.suptitle("4OCB molecular center-of-mass radial distributions", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def command_rdf(args: argparse.Namespace) -> None:
    runs = discover_runs(args.raw_root.resolve())
    topology = ensure_topology(runs, args.topology.resolve())
    cpptraj = resolve_cpptraj(args.cpptraj)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_rdf_run,
                run,
                str(topology),
                str(args.out_data.resolve()),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.spacing_a,
                args.maximum_a,
                args.force,
            ): run.label
            for run in runs
        }
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"Finished RDF: {futures[future]}", flush=True)
    rows.sort(key=lambda row: float(row["temperature_K"]))
    write_csv(args.out_data.resolve() / "summary.csv", rows)
    plot_rdf(rows, args.out_plot.resolve())


def calculate_pair_ranks(
    centers_nm: np.ndarray,
    boxes_nm: np.ndarray,
    axes: np.ndarray,
    maximum_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    include_g1: bool,
) -> tuple[
    dict[float, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[float, dict[str, np.ndarray]] | None,
    dict[str, np.ndarray] | None,
    np.ndarray,
]:
    """Accumulate g2 and optional g1 with one pair search per geometry/frame."""

    def empty_values(rank: int) -> dict[str, np.ndarray]:
        return {
            "count": np.zeros(nbins),
            f"p{rank}_sum": np.zeros(nbins),
            "g0_sum": np.zeros(nbins),
            f"g{rank}_sum": np.zeros(nbins),
        }

    parallel_g2 = {radius: empty_values(2) for radius in RADII_NM}
    perpendicular_g2 = empty_values(2)
    parallel_g1 = (
        {radius: empty_values(1) for radius in RADII_NM} if include_g1 else None
    )
    perpendicular_g1 = empty_values(1) if include_g1 else None
    s_values = np.empty(len(centers_nm))
    bin_width = maximum_nm / nbins
    radial_edges = np.linspace(0.0, maximum_nm, nbins + 1)
    annular_factor = (
        np.pi
        * slab_halfwidth_nm
        * (radial_edges[1:] ** 2 - radial_edges[:-1] ** 2)
    )

    def accumulate(
        values: dict[str, np.ndarray],
        coordinate: np.ndarray,
        weight: np.ndarray,
        selected: np.ndarray,
        norm: float | np.ndarray,
        rank: int,
    ) -> None:
        count, weighted = histogram_pair_values(
            coordinate, weight, selected, maximum_nm, nbins
        )
        values["count"] += count
        values[f"p{rank}_sum"] += weighted
        values["g0_sum"] += count / norm
        values[f"g{rank}_sum"] += weighted / norm

    for frame, (centers, box, frame_axes) in enumerate(
        zip(centers_nm, boxes_nm, axes, strict=True)
    ):
        director, s_values[frame] = director_from_axes(frame_axes)
        basis = director_basis(director)
        pair_density = NMOLECULES * (NMOLECULES - 1.0) / float(np.prod(box))

        first, second, displacement = candidate_pairs(
            centers,
            box,
            basis,
            np.asarray([max(RADII_NM), max(RADII_NM), maximum_nm]),
        )
        parallel_coordinate = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        p1 = np.einsum("ij,ij->i", frame_axes[first], frame_axes[second])
        p2 = 1.5 * p1 * p1 - 0.5
        for radius in RADII_NM:
            selected = (parallel_coordinate < maximum_nm) & (radial < radius)
            norm = pair_density * np.pi * radius * radius * bin_width
            accumulate(
                parallel_g2[radius], parallel_coordinate, p2, selected, norm, 2
            )
            if parallel_g1 is not None:
                accumulate(
                    parallel_g1[radius], parallel_coordinate, p1, selected, norm, 1
                )

        first, second, displacement = candidate_pairs(
            centers,
            box,
            basis,
            np.asarray([maximum_nm, maximum_nm, slab_halfwidth_nm]),
        )
        perpendicular_coordinate = np.hypot(displacement[:, 0], displacement[:, 1])
        selected = (
            (np.abs(displacement[:, 2]) < slab_halfwidth_nm)
            & (perpendicular_coordinate < maximum_nm)
        )
        p1 = np.einsum("ij,ij->i", frame_axes[first], frame_axes[second])
        p2 = 1.5 * p1 * p1 - 0.5
        norm = pair_density * annular_factor
        accumulate(
            perpendicular_g2, perpendicular_coordinate, p2, selected, norm, 2
        )
        if perpendicular_g1 is not None:
            accumulate(
                perpendicular_g1, perpendicular_coordinate, p1, selected, norm, 1
            )

        if (frame + 1) % 100 == 0 or frame + 1 == len(centers_nm):
            ranks = "g1/g2" if include_g1 else "g2"
            print(
                f"  processed {ranks} frame {frame + 1}/{len(centers_nm)}",
                flush=True,
            )

    def finalize(values: dict[str, np.ndarray], rank: int) -> None:
        values["g0"] = values.pop("g0_sum") / len(centers_nm)
        values[f"g{rank}"] = values.pop(f"g{rank}_sum") / len(centers_nm)
        values[f"C{rank}"] = np.divide(
            values[f"g{rank}"],
            values["g0"],
            out=np.full(nbins, np.nan),
            where=values["g0"] > 0.0,
        )

    for values in [*parallel_g2.values(), perpendicular_g2]:
        finalize(values, 2)
    if parallel_g1 is not None and perpendicular_g1 is not None:
        for values in [*parallel_g1.values(), perpendicular_g1]:
            finalize(values, 1)
    return (
        parallel_g2,
        perpendicular_g2,
        parallel_g1,
        perpendicular_g1,
        s_values,
    )


def process_paircorr_run(
    run: Run,
    topology: str,
    order_root: str,
    output_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    maximum_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force_cpptraj: bool,
    force_analysis: bool,
) -> dict[str, object]:
    output_root_path = Path(output_root)
    geometry = output_root_path / "geometry" / run.label
    geometry.mkdir(parents=True, exist_ok=True)
    center_path = geometry / "molecular_centers_and_box_cpptraj.dat"
    input_path = geometry / "molecular_centers_and_box.cpptraj.in"
    log_path = geometry / "molecular_centers_and_box.cpptraj.log"
    write_center_input(
        input_path, Path(topology), run.trajectory, center_path, start, stop, stride
    )
    reused_geometry = center_path.is_file() and not force_cpptraj
    if not reused_geometry:
        run_cpptraj(cpptraj, input_path, log_path)
    center_frames, centers, boxes = load_cpptraj_centers(center_path)

    order_run = Path(order_root) / run.label
    axis_arrays = {
        "end_to_end": np.load(order_run / "end_to_end" / "axes.npy"),
        "core": np.load(order_run / "core" / "axes.npy"),
        "inertia": np.load(order_run / "inertia" / "axes.npy"),
    }
    expected_frames = (stop - start) // stride + 1
    if len(center_frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} center frames for {run.label}")
    if any(len(axes) != expected_frames for axes in axis_arrays.values()):
        raise ValueError(
            f"Order products for {run.label} do not match this frame selection; "
            "rerun `order` with the same --start/--stop/--stride"
        )

    reused_all = True
    means: dict[str, float] = {}
    for axis_name, axes in axis_arrays.items():
        axis_output = output_root_path / axis_name / run.label
        axis_output.mkdir(parents=True, exist_ok=True)
        g2_paths = [
            axis_output / f"parallel_R{radius:.1f}nm_g2.dat" for radius in RADII_NM
        ] + [axis_output / "perpendicular_slab_g2.dat"]
        reuse_g2 = all(path.is_file() for path in g2_paths) and not force_analysis
        reused_all &= reuse_g2
        g1_paths = (
            [
                axis_output / f"parallel_R{radius:.1f}nm_g1.dat"
                for radius in RADII_NM
            ]
            + [axis_output / "perpendicular_slab_g1.dat"]
            if axis_name != "inertia"
            else []
        )
        reuse_g1 = (
            all(path.is_file() for path in g1_paths) and not force_analysis
            if g1_paths
            else True
        )
        reused_all &= reuse_g1
        if not reuse_g2 or not reuse_g1:
            (
                parallel_g2,
                perpendicular_g2,
                parallel_g1,
                perpendicular_g1,
                s_values,
            ) = calculate_pair_ranks(
                centers,
                boxes,
                axes,
                maximum_nm,
                slab_halfwidth_nm,
                nbins,
                include_g1=not reuse_g1,
            )
            z = (np.arange(nbins) + 0.5) * maximum_nm / nbins
            r = z.copy()
            if not reuse_g2:
                for radius, values in parallel_g2.items():
                    write_g2_table(
                        axis_output / f"parallel_R{radius:.1f}nm_g2.dat", z, values
                    )
                write_g2_table(
                    axis_output / "perpendicular_slab_g2.dat", r, perpendicular_g2
                )
                np.save(axis_output / "S_recomputed_by_frame.npy", s_values)
            if not reuse_g1 and parallel_g1 is not None and perpendicular_g1 is not None:
                for radius, values in parallel_g1.items():
                    write_g1_table(
                        axis_output / f"parallel_R{radius:.1f}nm_g1.dat", z, values
                    )
                write_g1_table(
                    axis_output / "perpendicular_slab_g1.dat", r, perpendicular_g1
                )
        else:
            s_values = np.load(order_run / axis_name / "S_by_frame.npy")
        means[axis_name] = float(s_values.mean())

    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "source_arc": str(run.trajectory),
        "frames": expected_frames,
        "molecules": NMOLECULES,
        "mean_S_end_to_end": means["end_to_end"],
        "mean_S_core": means["core"],
        "mean_S_inertia": means["inertia"],
        "cpptraj_geometry_reused": str(reused_geometry),
        "correlations_reused": str(reused_all),
    }


def mirror_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate((-x[::-1], x)), np.concatenate((y[::-1], y))


def plot_paircorr(
    rows: list[dict[str, object]],
    root: Path,
    output: Path,
    rank: int,
    perpendicular: bool,
) -> None:
    axis_names = ("end_to_end", "core") if rank == 1 else ("end_to_end", "core", "inertia")
    axis_styles = {
        "end_to_end": ("end-to-end", "-", None),
        "core": ("biphenyl core", "--", None),
        "inertia": ("inertia", ":", None),
    }
    colors = {1.0: "#d95f02", 2.0: "#1b9e77", 3.0: "#7570b3"}
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11.0), sharex=True)
    flat = axes.ravel()
    for panel, (ax, row) in enumerate(zip(flat, rows, strict=False)):
        run = str(row["run"])
        for axis_name in axis_names:
            label, linestyle, marker = axis_styles[axis_name]
            if perpendicular:
                path = root / axis_name / run / f"perpendicular_slab_g{rank}.dat"
                result = read_g1(path) if rank == 1 else read_result(path)
                ax.plot(
                    result["x"], result[f"g{rank}"], color="#377eb8",
                    linestyle=linestyle, linewidth=1.4, marker=marker,
                    label=label if panel == 0 else None,
                )
            else:
                for radius in RADII_NM:
                    path = root / axis_name / run / f"parallel_R{radius:.1f}nm_g{rank}.dat"
                    result = read_g1(path) if rank == 1 else read_result(path)
                    x, y = mirror_curve(result["x"], result[f"g{rank}"])
                    ax.plot(
                        x, y, color=colors[radius], linestyle=linestyle,
                        linewidth=1.25,
                        label=(f"R={radius:g} nm, {label}" if panel == 0 else None),
                    )
        ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.7)
        ax.set_title(f"{float(row['temperature_K']):.0f} K")
        ax.grid(alpha=0.16)
    flat[-1].axis("off")
    flat[0].legend(frameon=False, fontsize=7.2, ncol=1)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"weighted g{rank}")
    for ax in axes[-1, :2]:
        ax.set_xlabel("r_perp (nm)" if perpendicular else "r_parallel (nm)")
    geometry = "perpendicular to" if perpendicular else "parallel to"
    rank_label = "first-rank" if rank == 1 else "second-rank"
    fig.suptitle(
        f"4OCB {rank_label} director-resolved correlations\n"
        f"{geometry} the framewise director",
        fontsize=14,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.065,
        top=0.88,
        hspace=0.28,
        wspace=0.18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def command_paircorr(args: argparse.Namespace) -> None:
    runs = discover_runs(args.raw_root.resolve())
    topology = ensure_topology(runs, args.topology.resolve())
    cpptraj = resolve_cpptraj(args.cpptraj)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_paircorr_run,
                run,
                str(topology),
                str(args.order_root.resolve()),
                str(args.out_data.resolve()),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.maximum_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_cpptraj,
                args.force_analysis,
            ): run.label
            for run in runs
        }
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"Finished pair correlations: {futures[future]}", flush=True)
    rows.sort(key=lambda row: float(row["temperature_K"]))
    output = args.out_data.resolve()
    write_csv(output / "summary.csv", rows)
    report = args.out_report.resolve()
    plot_paircorr(rows, output, report / "parallel_g1_11panel.png", 1, False)
    plot_paircorr(rows, output, report / "perpendicular_g1_11panel.png", 1, True)
    plot_paircorr(rows, output, report / "parallel_g2_11panel.png", 2, False)
    plot_paircorr(rows, output, report / "perpendicular_g2_11panel.png", 2, True)


def global_atom(molecule: int, local_atom: int) -> int:
    return (molecule - 1) * ATOMS_PER_MOLECULE + local_atom


def write_dihedral_input(
    path: Path,
    topology: Path,
    run: Run,
    output: Path,
    start: int,
    stop: int,
    stride: int,
    dihedral_atoms: tuple[int, int, int, int],
    dataset_stem: str,
) -> None:
    lines = [
        f"parm {quote_cpptraj(topology)}",
        f"trajin {quote_cpptraj(run.trajectory)} {start} {stop} {stride}",
        "autoimage",
    ]
    names = []
    for molecule in range(1, NMOLECULES + 1):
        name = f"m{molecule:04d}_{dataset_stem}"
        names.append(name)
        masks = " ".join(
            f"@{global_atom(molecule, atom)}" for atom in dihedral_atoms
        )
        lines.append(f"dihedral {name} {masks}")
    lines.extend(
        [
            "run",
            f"writedata {quote_cpptraj(output)} {' '.join(names)} prec 12.6",
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def angle_histogram(angles: np.ndarray, temperature: float) -> np.ndarray:
    edges = np.linspace(0.0, 360.0, 361)
    density, _ = np.histogram(angles.ravel() % 360.0, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pmf = np.full_like(density, np.nan)
    positive = density > 0.0
    pmf[positive] = -R_KCAL * temperature * np.log(density[positive])
    pmf[positive] -= np.nanmin(pmf)
    return np.column_stack((centers, density, pmf))


def process_dihedral_run(
    run: Run,
    topology: str,
    output_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    force: bool,
    dihedral_atoms: tuple[int, int, int, int],
    output_stem: str,
) -> dict[str, object]:
    output = Path(output_root) / run.label
    output.mkdir(parents=True, exist_ok=True)
    raw = output / f"{output_stem}_dihedrals_cpptraj.dat"
    input_path = output / f"{output_stem}_dihedrals.cpptraj.in"
    log = output / f"{output_stem}_dihedrals.cpptraj.log"
    write_dihedral_input(
        input_path,
        Path(topology),
        run,
        raw,
        start,
        stop,
        stride,
        dihedral_atoms,
        output_stem,
    )
    reused = raw.is_file() and not force
    if not reused:
        run_cpptraj(cpptraj, input_path, log)
    table = np.loadtxt(raw, comments="#", ndmin=2)
    if table.shape[1] != NMOLECULES + 1:
        raise ValueError(f"Expected {NMOLECULES + 1} columns in {raw}")
    frames = table[:, 0].astype(int)
    angles = table[:, 1:]
    np.save(output / f"{output_stem}_dihedrals_deg.npy", angles)
    histogram = angle_histogram(angles, run.temperature)
    histogram_path = output / f"{output_stem}_dihedral_histogram.csv"
    np.savetxt(
        histogram_path,
        histogram,
        delimiter=",",
        header="angle_center_deg,P_per_degree,PMF_kcal_per_mol_shifted_min_zero",
        comments="",
        fmt="%.10e",
    )
    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "source_arc": str(run.trajectory),
        "frames": len(frames),
        "molecules": NMOLECULES,
        "samples": angles.size,
        "mode_deg_0_360": float(histogram[np.argmax(histogram[:, 1]), 0]),
        "histogram_csv": str(histogram_path),
        "cpptraj_reused": str(reused),
    }


def plot_dihedral(
    rows: list[dict[str, object]],
    output: Path,
    column: int,
    xlabel: str,
    title: str,
    symbol: str,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 10.5), sharex=True)
    cmap = plt.get_cmap("turbo", len(rows))
    for index, (ax, row) in enumerate(zip(axes.ravel(), rows, strict=False)):
        table = np.loadtxt(str(row["histogram_csv"]), delimiter=",", skiprows=1)
        ax.plot(table[:, 0], table[:, column], color=cmap(index), linewidth=1.4)
        ax.set_title(f"{float(row['temperature_K']):.0f} K")
        ax.grid(alpha=0.17)
    axes.ravel()[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"P({symbol}) per degree" if column == 1 else "PMF (kcal/mol)")
    for ax in axes[-1, :2]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"{title} probability" if column == 1 else f"{title} PMF", fontsize=15)
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.065,
        top=0.92,
        hspace=0.28,
        wspace=0.18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def command_dihedral(args: argparse.Namespace) -> None:
    runs = discover_runs(args.raw_root.resolve())
    topology = ensure_topology(runs, args.topology.resolve())
    cpptraj = resolve_cpptraj(args.cpptraj)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_dihedral_run,
                run,
                str(topology),
                str(args.out_data.resolve()),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.force,
                args.dihedral_atoms,
                args.output_stem,
            ): run.label
            for run in runs
        }
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"Finished dihedral: {futures[future]}", flush=True)
    rows.sort(key=lambda row: float(row["temperature_K"]))
    output = args.out_data.resolve()
    write_csv(output / "summary.csv", rows)
    report = args.out_report.resolve()
    plot_dihedral(
        rows,
        report / args.probability_figure,
        1,
        args.xlabel,
        args.title,
        args.symbol,
    )
    plot_dihedral(
        rows,
        report / args.pmf_figure,
        2,
        args.xlabel,
        args.title,
        args.symbol,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=DATASET.raw_path())
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(
            "data-processed/4ocb/amoeba-pt2.2/topology/4ocb-x1000-first-frame.pdb"
        ),
    )
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    order = subparsers.add_parser("order", help="Calculate three-axis nematic order")
    add_common_arguments(order)
    order.add_argument(
        "--out-data",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/order"),
    )
    order.add_argument(
        "--out-plot",
        type=Path,
        default=Path("reports/4ocb_order/s_vs_temperature_three_axes.png"),
    )
    order.add_argument("--force", action="store_true")
    order.set_defaults(handler=command_order)

    rdf = subparsers.add_parser("rdf", help="Calculate spherical molecular COM RDFs")
    add_common_arguments(rdf)
    rdf.add_argument(
        "--out-data",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/com-rdf"),
    )
    rdf.add_argument(
        "--out-plot",
        type=Path,
        default=Path("reports/4ocb_com_rdf/com_com_rdf_11panel.png"),
    )
    rdf.add_argument("--spacing-A", dest="spacing_a", type=float, default=0.1)
    rdf.add_argument("--maximum-A", dest="maximum_a", type=float, default=35.0)
    rdf.add_argument("--force", action="store_true")
    rdf.set_defaults(handler=command_rdf)

    paircorr = subparsers.add_parser(
        "paircorr", help="Calculate end-to-end/core/inertia director correlations"
    )
    add_common_arguments(paircorr)
    paircorr.add_argument(
        "--order-root",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/order"),
    )
    paircorr.add_argument(
        "--out-data",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/director-paircorr"),
    )
    paircorr.add_argument(
        "--out-report", type=Path, default=Path("reports/4ocb_director_paircorr")
    )
    paircorr.add_argument("--maximum-nm", type=float, default=5.0)
    paircorr.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    paircorr.add_argument("--nbins", type=int, default=200)
    paircorr.add_argument("--force-cpptraj", action="store_true")
    paircorr.add_argument("--force-analysis", action="store_true")
    paircorr.set_defaults(handler=command_paircorr)

    dihedral = subparsers.add_parser(
        "dihedral", help="Calculate the 16-15-28-11 CC-O-C_ar torsion"
    )
    add_common_arguments(dihedral)
    dihedral.add_argument(
        "--out-data",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/junction-dihedral"),
    )
    dihedral.add_argument(
        "--out-report", type=Path, default=Path("reports/4ocb_junction_dihedral")
    )
    dihedral.add_argument("--force", action="store_true")
    dihedral.set_defaults(
        handler=command_dihedral,
        dihedral_atoms=ETHER_DIHEDRAL,
        output_stem="junction",
        probability_figure="P_chi_CC_O_Car_11panel.png",
        pmf_figure="PMF_chi_CC_O_Car_11panel.png",
        xlabel="C-C-O-C(ar) dihedral chi (degrees)",
        title="4OCB C-C-O-C(ar) ether torsion",
        symbol="chi",
    )

    ring_dihedral = subparsers.add_parser(
        "ring-dihedral",
        help="Calculate the 10-11-28-15 ring-plane junction torsion",
    )
    add_common_arguments(ring_dihedral)
    ring_dihedral.add_argument(
        "--out-data",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2/ring-junction-dihedral"),
    )
    ring_dihedral.add_argument(
        "--out-report",
        type=Path,
        default=Path("reports/4ocb_ring_junction_dihedral"),
    )
    ring_dihedral.add_argument("--force", action="store_true")
    ring_dihedral.set_defaults(
        handler=command_dihedral,
        dihedral_atoms=RING_JUNCTION_DIHEDRAL,
        output_stem="ring_junction",
        probability_figure="P_phi_Cring_Cipso_O_C_11panel.png",
        pmf_figure="PMF_phi_Cring_Cipso_O_C_11panel.png",
        xlabel="C(ring)-C(ipso)-O-C dihedral phi (degrees)",
        title="4OCB aryl-O ring-junction torsion",
        symbol="phi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.start <= args.stop <= DATASET.frames:
        raise SystemExit(f"Require 1 <= --start <= --stop <= {DATASET.frames}")
    if args.stride < 1 or args.workers < 1:
        raise SystemExit("--stride and --workers must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()


