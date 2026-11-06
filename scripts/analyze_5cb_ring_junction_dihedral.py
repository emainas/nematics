#!/usr/bin/env python3
"""Calculate the 5CB ring-to-pentyl junction dihedral for all current runs.

The dihedral is C09-C10-C14-C15 (one-based local atoms 8-10-15-16),
centered on the bond between ring ipso carbon C10 and first pentyl carbon C14.
AMOEBA ARC trajectories are evaluated with cpptraj. GAFF GRO snapshots are
read without modification and evaluated with the same signed convention.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nematics.gro import read_gro_frame  # noqa: E402


NMOLECULES = 1000
ATOMS_PER_MOLECULE = 38
DIHEDRAL = (8, 10, 15, 16)  # one-based: C09-C10-C14-C15
GAFF_RUNS = (("gro", 356.0), ("gro2", 366.0), ("gro3", 376.0), ("gro4", 386.0))
R_KCAL = 0.00198720425864083


@dataclass(frozen=True)
class AmoebaRun:
    label: str
    temperature: float
    trajectory: Path


def discover_amoeba(root: Path) -> list[AmoebaRun]:
    runs: list[AmoebaRun] = []
    for trajectory in sorted(root.glob("*/5cb-x1000.arc")):
        match = re.search(r"-(\d{3})$", trajectory.parent.name)
        if match is None:
            raise ValueError(f"Cannot infer temperature from {trajectory.parent.name}")
        runs.append(
            AmoebaRun(trajectory.parent.name, float(match.group(1)), trajectory.resolve())
        )
    if len(runs) != 11:
        raise ValueError(f"Expected 11 AMOEBA runs, found {len(runs)}")
    return sorted(runs, key=lambda run: run.temperature)


def quote_cpptraj(path: Path) -> str:
    value = str(path)
    if '"' in value:
        raise ValueError(f"Unsupported quote in path: {path}")
    return f'"{value}"'


def global_atom(molecule: int, local_atom: int) -> int:
    return (molecule - 1) * ATOMS_PER_MOLECULE + local_atom


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    output: Path,
    start: int,
    stop: int,
    stride: int,
) -> None:
    lines = [
        f"parm {quote_cpptraj(topology)}",
        f"trajin {quote_cpptraj(trajectory)} {start} {stop} {stride}",
        "autoimage",
    ]
    names = []
    for molecule in range(1, NMOLECULES + 1):
        name = f"m{molecule:04d}_ring_junction"
        names.append(name)
        masks = " ".join(f"@{global_atom(molecule, atom)}" for atom in DIHEDRAL)
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


def run_cpptraj(executable: str, input_path: Path, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [executable, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def load_cpptraj_angles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] != NMOLECULES + 1:
        raise ValueError(f"Expected {NMOLECULES + 1} columns in {path}, got {table.shape[1]}")
    return table[:, 0].astype(int), table[:, 1:]


def analyze_amoeba(
    run: AmoebaRun,
    topology: str,
    output_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    force: bool,
) -> tuple[dict[str, str | int | float], np.ndarray]:
    output = Path(output_root) / "amoeba" / run.label
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "ring_junction_dihedrals_cpptraj.dat"
    input_path = output / "ring_junction_dihedrals.cpptraj.in"
    log_path = output / "ring_junction_dihedrals.cpptraj.log"
    write_cpptraj_input(
        input_path, Path(topology), run.trajectory, raw_path, start, stop, stride
    )
    reused = raw_path.is_file() and not force
    if not reused:
        run_cpptraj(cpptraj, input_path, log_path)
    frames, angles = load_cpptraj_angles(raw_path)
    np.save(output / "ring_junction_dihedrals_deg.npy", angles)
    row = summarize("AMOEBA", run.label, run.temperature, frames, angles)
    row["source"] = str(run.trajectory)
    row["raw_reused"] = str(reused)
    return row, angles


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)\.gro", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def minimum_image(vector: np.ndarray, box: np.ndarray) -> np.ndarray:
    return vector - box * np.rint(vector / box)


def signed_dihedral(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return signed degrees for points shaped (molecules, 4, 3)."""
    b0 = -minimum_image(points[:, 1] - points[:, 0], box)
    b1 = minimum_image(points[:, 2] - points[:, 1], box)
    b2 = minimum_image(points[:, 3] - points[:, 2], box)
    b1 /= np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.einsum("mi,mi->m", b0, b1)[:, None] * b1
    w = b2 - np.einsum("mi,mi->m", b2, b1)[:, None] * b1
    x = np.einsum("mi,mi->m", v, w)
    y = np.einsum("mi,mi->m", np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def analyze_gaff(
    run: str,
    temperature: float,
    source_root: str,
    output_root: str,
    force: bool,
) -> tuple[dict[str, str | int | float], np.ndarray]:
    source = Path(source_root) / run
    output = Path(output_root) / "gaff" / run
    output.mkdir(parents=True, exist_ok=True)
    angle_path = output / "ring_junction_dihedrals_deg.npy"
    paths = sorted(source.glob("frame*.gro"), key=frame_number)
    if not paths:
        raise FileNotFoundError(source)
    reused = angle_path.is_file() and not force
    if reused:
        angles = np.load(angle_path)
    else:
        angles = np.empty((len(paths), NMOLECULES), dtype=float)
        indices = np.asarray(DIHEDRAL) - 1
        for frame_index, path in enumerate(paths):
            frame = read_gro_frame(path, natoms_per_mol=ATOMS_PER_MOLECULE)
            angles[frame_index] = signed_dihedral(frame.positions[:, indices, :], frame.box)
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == len(paths):
                print(f"  {run}: dihedral frame {frame_index + 1}/{len(paths)}", flush=True)
        np.save(angle_path, angles)
    frames = np.asarray([frame_number(path) for path in paths])
    row = summarize("GAFF", run, temperature, frames, angles)
    row["source"] = str(source.resolve())
    row["raw_reused"] = str(reused)
    return row, angles


def histogram(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 360.0, 361)
    density, _ = np.histogram(angles.ravel() % 360.0, bins=edges, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), density


def summarize(
    force_field: str,
    run: str,
    temperature: float,
    frames: np.ndarray,
    angles: np.ndarray,
) -> dict[str, str | int | float]:
    centers, density = histogram(angles)
    radians = np.radians(angles.ravel())
    mean_vector = np.mean(np.exp(1j * radians))
    wrapped = angles.ravel() % 360.0
    distance_to_planar = np.minimum.reduce(
        (wrapped, np.abs(wrapped - 180.0), 360.0 - wrapped)
    )
    return {
        "force_field": force_field,
        "run": run,
        "temperature_K": temperature,
        "frames": angles.shape[0],
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "molecules": angles.shape[1],
        "samples": angles.size,
        "mode_deg_0_360": float(centers[np.argmax(density)]),
        "circular_mean_deg_0_360": float(np.degrees(np.angle(mean_vector)) % 360.0),
        "circular_resultant": float(abs(mean_vector)),
        "mean_distance_to_coplanar_deg": float(distance_to_planar.mean()),
        "fraction_within_30deg_of_coplanar": float(np.mean(distance_to_planar <= 30.0)),
    }


def write_histogram(path: Path, angles: np.ndarray, temperature: float) -> None:
    centers, density = histogram(angles)
    probability = density.copy()
    positive = probability > 0.0
    pmf = np.full_like(probability, np.nan)
    pmf[positive] = -R_KCAL * temperature * np.log(probability[positive])
    pmf[positive] -= np.nanmin(pmf)
    np.savetxt(
        path,
        np.column_stack((centers, density, pmf)),
        delimiter=",",
        header="chi_center_deg,P_chi_per_degree,PMF_kcal_per_mol_shifted_min_zero",
        comments="",
        fmt="%.10e",
    )


def plot_panels(
    records: list[tuple[dict[str, str | int | float], np.ndarray]],
    output_root: Path,
) -> None:
    records = sorted(records, key=lambda item: float(item[0]["temperature_K"]))
    cmap = plt.get_cmap("turbo", len(records))
    for quantity in ("density", "pmf"):
        fig, axes = plt.subplots(
            4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True
        )
        for index, (row, angles) in enumerate(records):
            centers, density = histogram(angles)
            if quantity == "density":
                values = density
            else:
                values = np.full_like(density, np.nan)
                positive = density > 0.0
                temperature = float(row["temperature_K"])
                values[positive] = -R_KCAL * temperature * np.log(density[positive])
                values[positive] -= np.nanmin(values)
            ax = axes.flat[index]
            ax.plot(centers, values, color=cmap(index), lw=1.3)
            ax.axvline(180.0, color="0.45", ls=":", lw=0.75)
            ax.set_title(
                f"{float(row['temperature_K']):.0f} K · {row['force_field']}", fontsize=10
            )
            ax.grid(alpha=0.16)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(
            0.05,
            0.65,
            "C09–C10–C14–C15\nlocal atoms 8–10–15–16\ncentral bond C10–C14",
            transform=axes.flat[-1].transAxes,
            fontsize=11,
            va="top",
        )
        for ax in axes[:, 0]:
            ax.set_ylabel("P(χ) per degree" if quantity == "density" else "PMF (kcal/mol)")
        for ax in axes[-1, :3]:
            ax.set_xlabel("χ ring junction (degrees)")
        for ax in axes.flat[:-1]:
            ax.set_xlim(0.0, 360.0)
            ax.set_xticks((0, 90, 180, 270, 360))
        title = (
            "5CB ring-to-pentyl junction dihedral distributions"
            if quantity == "density"
            else "5CB ring-to-pentyl junction dihedral PMFs"
        )
        fig.suptitle(title, fontsize=14)
        name = "P_chi_ring_junction_15panel.png" if quantity == "density" else "PMF_chi_ring_junction_15panel.png"
        fig.savefig(output_root / name, dpi=300)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amoeba-root",
        type=Path,
        default=Path("/hickory/proj/cdpoonlab/users/emainas/tinker9-5cb-x1000-pt2.7"),
    )
    parser.add_argument(
        "--gaff-root",
        type=Path,
        default=Path("/hickory/proj/cdpoonlab/users/emainas/5cb-x1000-gaff-88x88x94-zalign"),
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/topology/5cb-x1000-first-frame.pdb"),
    )
    parser.add_argument(
        "--out-data", type=Path, default=Path("data-processed/5cb/comparisons/ring-junction-dihedral")
    )
    parser.add_argument(
        "--out-report", type=Path, default=Path("reports/5cb_ring_junction_dihedral")
    )
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpptraj = shutil.which(args.cpptraj)
    if cpptraj is None:
        raise SystemExit(f"cpptraj executable not found: {args.cpptraj}")
    amoeba_runs = discover_amoeba(args.amoeba_root.resolve())
    output_data = args.out_data.resolve()
    output_report = args.out_report.resolve()
    output_data.mkdir(parents=True, exist_ok=True)
    output_report.mkdir(parents=True, exist_ok=True)

    records: list[tuple[dict[str, str | int | float], np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for run in amoeba_runs:
            future = executor.submit(
                analyze_amoeba,
                run,
                str(args.topology.resolve()),
                str(output_data),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.force,
            )
            futures[future] = f"AMOEBA/{run.label}"
        for run, temperature in GAFF_RUNS:
            future = executor.submit(
                analyze_gaff,
                run,
                temperature,
                str(args.gaff_root.resolve()),
                str(output_data),
                args.force,
            )
            futures[future] = f"GAFF/{run}"
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"Finished {futures[future]}", flush=True)

    records.sort(key=lambda item: float(item[0]["temperature_K"]))
    rows = [item[0] for item in records]
    with (output_data / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row, angles in records:
        subdir = "amoeba" if row["force_field"] == "AMOEBA" else "gaff"
        write_histogram(
            output_data / subdir / str(row["run"]) / "ring_junction_histogram.csv",
            angles,
            float(row["temperature_K"]),
        )
    plot_panels(records, output_report)

    (output_data / "README.md").write_text(
        "# 5CB ring-to-pentyl junction dihedral\n\n"
        "The adopted torsion is `C09-C10-C14-C15`, one-based local atoms "
        "`8-10-15-16`, centered on the ring-ipso C10--first-pentyl-C14 bond. "
        "C09 is the ortho ring carbon establishing the aromatic plane. Choosing "
        "the other ortho carbon C11 would describe the same physical rotation "
        "with an approximately 180-degree-shifted convention.\n\n"
        "AMOEBA was evaluated by cpptraj after `autoimage`; every run retains "
        "the cpptraj input, log, raw DAT table, and NPY angle matrix. GAFF was "
        "evaluated from the read-only GRO snapshots with sequential minimum-image "
        "bond vectors. Each trajectory contributes 1,000 frames x 1,000 molecules. "
        "Histograms use 360 one-degree bins on [0,360). PMFs are "
        "`-R T ln P + C` in kcal/mol with each minimum shifted to zero. The GAFF "
        "temperature mapping remains inferred from the legacy report.\n"
    )
    print(f"Wrote {output_data / 'summary.csv'}")
    print(f"Wrote figures beneath {output_report}")


if __name__ == "__main__":
    main()
