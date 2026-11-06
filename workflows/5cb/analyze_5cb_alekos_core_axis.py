#!/usr/bin/env python3
"""5CB order and pair correlations using Alekos's C10-to-C04 core axis.

The chemically directed axis is unit(r_C04-r_C10), i.e. local atom 10 to
local atom 9.  The same axes define each frame's Q-tensor director and the
P1/P2 pair weights.  AMOEBA axes are extracted by cpptraj after autoimage;
GAFF axes are built from read-only GRO frames with minimum-image displacement.
Existing repository-local molecular COM and box caches are reused.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_5cb_g1_director_paircorr import alekos_g1  # noqa: E402
from analyze_amoeba_5cb_director_paircorr import (  # noqa: E402
    NMOLECULES,
    RADII_NM,
    alekos_curve,
    candidate_pairs,
    director_basis,
    load_cpptraj_axes,
    load_cpptraj_centers,
    mirror_parallel_curve,
    read_csv_records,
    read_result,
)
from nematics.datasets import get_dataset  # noqa: E402
from nematics.gro import read_gro_frame  # noqa: E402


NATOMS_PER_MOLECULE = 38
C04 = 8  # zero-based local atom 9
C10 = 9  # zero-based local atom 10
GAFF_5CB = get_dataset("gaff-5cb")
AMOEBA_5CB = get_dataset("amoeba-5cb")
GAFF_RUNS = tuple((run.id, run.temperature_k) for run in GAFF_5CB.runs)


def quote_cpptraj(path: Path) -> str:
    value = str(path)
    if '"' in value:
        raise ValueError(f"Unsupported quote in cpptraj path: {path}")
    return f'"{value}"'


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    output: Path,
    nframes: int,
) -> None:
    path.write_text(
        "\n".join(
            (
                f"parm {quote_cpptraj(topology)}",
                f"trajin {quote_cpptraj(trajectory)} 1 {nframes} 1",
                "autoimage",
                "multivector core resrange 1-1000 name1 C10 name2 C04 "
                f"out {quote_cpptraj(output)}",
                f"precision {quote_cpptraj(output)} 13 6",
                "run",
                "quit",
                "",
            )
        )
    )


def run_cpptraj(executable: str, input_path: Path, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [executable, "-i", str(input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)\.gro", path.name)
    if match is None:
        raise ValueError(f"Not a numbered GRO frame: {path}")
    return int(match.group(1))


def frame_order(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = axes / np.linalg.norm(axes, axis=2)[:, :, None]
    q = 1.5 * np.einsum("fmi,fmj->fij", unit, unit) / unit.shape[1]
    q -= 0.5 * np.eye(3)[None, :, :]
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    return eigenvalues[:, -1], eigenvectors[:, :, -1], q


def histogram_weights(
    coordinate: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    selected: np.ndarray,
    maximum: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.floor(coordinate[selected] * nbins / maximum).astype(np.int64)
    valid = (indices >= 0) & (indices < nbins)
    indices = indices[valid]
    return (
        np.bincount(indices, minlength=nbins).astype(float),
        np.bincount(indices, weights=p1[selected][valid], minlength=nbins),
        np.bincount(indices, weights=p2[selected][valid], minlength=nbins),
    )


def calculate_correlations(
    centers: np.ndarray,
    boxes: np.ndarray,
    axes: np.ndarray,
    maximum: float,
    slab_halfwidth: float,
    nbins: int,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    nframes = centers.shape[0]
    fields = ("count", "p1_sum", "p2_sum", "g0_sum", "g1_sum", "g2_sum")
    parallel = {
        radius: {field: np.zeros(nbins) for field in fields} for radius in RADII_NM
    }
    perpendicular = {field: np.zeros(nbins) for field in fields}
    dz = maximum / nbins
    edges = np.linspace(0.0, maximum, nbins + 1)
    annular_factor = np.pi * slab_halfwidth * (edges[1:] ** 2 - edges[:-1] ** 2)

    for frame in range(nframes):
        _, directors, _ = frame_order(axes[frame : frame + 1])
        basis = director_basis(directors[0])
        pair_density = NMOLECULES * (NMOLECULES - 1.0) / float(np.prod(boxes[frame]))

        first, second, displacement = candidate_pairs(
            centers[frame],
            boxes[frame],
            basis,
            np.asarray([max(RADII_NM), max(RADII_NM), maximum]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        p1 = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        p2 = 1.5 * p1 * p1 - 0.5
        for radius in RADII_NM:
            selected = (z < maximum) & (radial < radius)
            count, p1_sum, p2_sum = histogram_weights(
                z, p1, p2, selected, maximum, nbins
            )
            norm = pair_density * np.pi * radius * radius * dz
            values = parallel[radius]
            values["count"] += count
            values["p1_sum"] += p1_sum
            values["p2_sum"] += p2_sum
            values["g0_sum"] += count / norm
            values["g1_sum"] += p1_sum / norm
            values["g2_sum"] += p2_sum / norm

        first, second, displacement = candidate_pairs(
            centers[frame],
            boxes[frame],
            basis,
            np.asarray([maximum, maximum, slab_halfwidth]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        p1 = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        p2 = 1.5 * p1 * p1 - 0.5
        selected = (z < slab_halfwidth) & (radial < maximum)
        count, p1_sum, p2_sum = histogram_weights(
            radial, p1, p2, selected, maximum, nbins
        )
        values = perpendicular
        values["count"] += count
        values["p1_sum"] += p1_sum
        values["p2_sum"] += p2_sum
        values["g0_sum"] += count / (pair_density * annular_factor)
        values["g1_sum"] += p1_sum / (pair_density * annular_factor)
        values["g2_sum"] += p2_sum / (pair_density * annular_factor)

        if (frame + 1) % 100 == 0 or frame + 1 == nframes:
            print(f"  processed core-axis frame {frame + 1}/{nframes}", flush=True)

    for values in list(parallel.values()) + [perpendicular]:
        values["g0"] = values.pop("g0_sum") / nframes
        values["g1"] = values.pop("g1_sum") / nframes
        values["g2"] = values.pop("g2_sum") / nframes
        values["C1"] = np.divide(
            values["g1"], values["g0"], out=np.full(nbins, np.nan), where=values["g0"] > 0
        )
        values["C2"] = np.divide(
            values["g2"], values["g0"], out=np.full(nbins, np.nan), where=values["g0"] > 0
        )
    return parallel, perpendicular


def write_table(path: Path, coordinate: np.ndarray, values: dict[str, np.ndarray]) -> None:
    np.savetxt(
        path,
        np.column_stack(
            (
                coordinate,
                values["count"],
                values["p1_sum"],
                values["p2_sum"],
                values["g0"],
                values["g1"],
                values["g2"],
                values["C1"],
                values["C2"],
            )
        ),
        header=(
            "coordinate_nm raw_pair_count raw_P1_sum raw_P2_sum g0 "
            "g1_P1_weighted g2_P2_weighted C1_equals_g1_over_g0 C2_equals_g2_over_g0"
        ),
        fmt=("%.8f", "%.0f", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"),
    )


def read_core(path: Path) -> dict[str, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    return {
        "x": table[:, 0],
        "count": table[:, 1],
        "g0": table[:, 4],
        "g1": table[:, 5],
        "g2": table[:, 6],
        "C1": table[:, 7],
        "C2": table[:, 8],
    }


def save_order(output: Path, frames: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_values, directors, q = frame_order(axes)
    np.save(output / "core_axes.npy", axes)
    np.save(output / "S_by_frame.npy", s_values)
    np.save(output / "director_by_frame.npy", directors)
    np.save(output / "Q_by_frame.npy", q)
    np.savetxt(
        output / "S_by_frame.csv",
        np.column_stack((frames, s_values, directors, np.trace(q, axis1=1, axis2=2))),
        delimiter=",",
        header="frame,S,n_x,n_y,n_z,Tr_Q",
        comments="",
        fmt=("%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.12e"),
    )
    return s_values, directors


def write_correlations(
    output: Path,
    centers: np.ndarray,
    boxes: np.ndarray,
    axes: np.ndarray,
    maximum: float,
    slab_halfwidth: float,
    nbins: int,
    force: bool,
) -> bool:
    expected = [output / f"parallel_R{r:.1f}nm.dat" for r in RADII_NM]
    expected.append(output / "perpendicular_slab.dat")
    reused = all(path.is_file() for path in expected) and not force
    if reused:
        return True
    parallel, perpendicular = calculate_correlations(
        centers, boxes, axes, maximum, slab_halfwidth, nbins
    )
    coordinate = (np.arange(nbins) + 0.5) * maximum / nbins
    for radius, values in parallel.items():
        write_table(output / f"parallel_R{radius:.1f}nm.dat", coordinate, values)
    write_table(output / "perpendicular_slab.dat", coordinate, perpendicular)
    return False


def process_amoeba(
    run: str,
    temperature: float,
    trajectory: str,
    topology: str,
    geometry_root: str,
    output_root: str,
    cpptraj: str,
    nframes: int,
    maximum: float,
    slab_halfwidth: float,
    nbins: int,
    force_axes: bool,
    force_correlations: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / "amoeba" / run
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "core_axis_vectors.dat"
    input_path = output / "core_axis_vectors.cpptraj.in"
    log_path = output / "core_axis_vectors.cpptraj.log"
    write_cpptraj_input(input_path, Path(topology), Path(trajectory), raw, nframes)
    reused_axes = raw.is_file() and not force_axes
    if not reused_axes:
        run_cpptraj(cpptraj, input_path, log_path)
    vector_frames, axes = load_cpptraj_axes(raw)
    vector_frames = vector_frames[:nframes]
    axes = axes[:nframes]
    center_frames, centers, boxes = load_cpptraj_centers(
        Path(geometry_root) / run / "molecular_centers_and_box_cpptraj.dat"
    )
    center_frames = center_frames[:nframes]
    centers = centers[:nframes]
    boxes = boxes[:nframes]
    if not np.array_equal(vector_frames, center_frames):
        raise ValueError(f"Frame mismatch for AMOEBA/{run}")
    s_values, _ = save_order(output, vector_frames, axes)
    reused_corr = write_correlations(
        output, centers, boxes, axes, maximum, slab_halfwidth, nbins, force_correlations
    )
    std = float(s_values.std(ddof=1))
    return {
        "force_field": "AMOEBA",
        "run": run,
        "temperature_K": temperature,
        "frames": nframes,
        "molecules": NMOLECULES,
        "axis": "C10_to_C04_Alekos_core",
        "mean_S": float(s_values.mean()),
        "std_S": std,
        "sem_S": std / np.sqrt(nframes),
        "source": trajectory,
        "axis_extraction": "cpptraj_autoimage_multivector_C10_to_C04",
        "axes_reused": str(reused_axes),
        "correlations_reused": str(reused_corr),
    }


def gaff_core_axes(
    source: Path,
    frame_numbers: np.ndarray,
    boxes: np.ndarray,
    output: Path,
    force: bool,
) -> tuple[np.ndarray, bool]:
    path = output / "core_axes.npy"
    if path.is_file() and not force:
        return np.load(path), True
    lookup = {frame_number(item): item for item in source.glob("frame*.gro")}
    axes = np.empty((len(frame_numbers), NMOLECULES, 3), dtype=float)
    for index, number in enumerate(frame_numbers):
        frame = read_gro_frame(lookup[int(number)], natoms_per_mol=NATOMS_PER_MOLECULE)
        displacement = frame.positions[:, C04] - frame.positions[:, C10]
        displacement -= boxes[index][None, :] * np.rint(displacement / boxes[index][None, :])
        axes[index] = displacement / np.linalg.norm(displacement, axis=1)[:, None]
        if (index + 1) % 100 == 0 or index + 1 == len(frame_numbers):
            print(f"  {source.name}: core-axis frame {index + 1}/{len(frame_numbers)}", flush=True)
    np.save(path, axes)
    return axes, False


def process_gaff(
    run: str,
    temperature: float,
    source_root: str,
    geometry_root: str,
    output_root: str,
    nframes: int,
    maximum: float,
    slab_halfwidth: float,
    nbins: int,
    force_axes: bool,
    force_correlations: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / "gaff" / run
    output.mkdir(parents=True, exist_ok=True)
    geometry = np.load(Path(geometry_root) / run / "molecular_geometry.npz")
    frames = geometry["frame_numbers"][:nframes]
    centers = geometry["centers_nm"][:nframes]
    boxes = geometry["boxes_nm"][:nframes]
    axes, reused_axes = gaff_core_axes(
        Path(source_root) / run, frames, boxes, output, force_axes
    )
    axes = axes[:nframes]
    s_values, _ = save_order(output, frames, axes)
    reused_corr = write_correlations(
        output, centers, boxes, axes, maximum, slab_halfwidth, nbins, force_correlations
    )
    std = float(s_values.std(ddof=1))
    return {
        "force_field": "GAFF",
        "run": run,
        "temperature_K": temperature,
        "frames": nframes,
        "molecules": NMOLECULES,
        "axis": "C10_to_C04_Alekos_core",
        "mean_S": float(s_values.mean()),
        "std_S": std,
        "sem_S": std / np.sqrt(nframes),
        "source": str((Path(source_root) / run).resolve()),
        "axis_extraction": "GRO_minimum_image_C10_to_C04",
        "axes_reused": str(reused_axes),
        "correlations_reused": str(reused_corr),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value_map(rows: list[dict[str, str]], value: str) -> dict[float, tuple[float, float]]:
    result = {}
    for row in rows:
        sem_key = "sem_S" if "sem_S" in row else None
        result[float(row["temperature_K"])] = (
            float(row[value]),
            float(row[sem_key]) if sem_key and row[sem_key] else 0.0,
        )
    return result


def plot_order(core_rows: list[dict[str, str]], output: Path) -> None:
    amoeba_end = value_map(read_csv(Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/summary.csv")), "mean_S")
    amoeba_inertia = value_map(read_csv(Path("data-processed/5cb/amoeba-pt2.7/inertia-order/summary.csv")), "mean_S")
    gaff_end = value_map(read_csv(Path("data-processed/5cb/gaff-zalign/end-to-end-order/summary.csv")), "mean_S")
    inertia_all = read_csv(Path("data-processed/5cb/comparisons/inertia-director-paircorr/summary.csv"))
    gaff_inertia = {
        float(row["temperature_K"]): (float(row["mean_S_inertia"]), 0.0)
        for row in inertia_all
        if row["force_field"] == "GAFF"
    }
    core = {
        row["force_field"]: {
            float(item["temperature_K"]): (float(item["mean_S"]), float(item["sem_S"]))
            for item in core_rows
            if item["force_field"] == row["force_field"]
        }
        for row in core_rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    styles = (
        ("End-to-end C18→N01", "#d95f02", "o"),
        ("Inertia, smallest moment", "#1f9e89", "s"),
        ("Alekos core C10→C04", "#5e3c99", "^"),
    )
    for ax, title, sources in (
        (axes[0], "AMOEBA", (amoeba_end, amoeba_inertia, core["AMOEBA"])),
        (axes[1], "GAFF (temperatures inferred)", (gaff_end, gaff_inertia, core["GAFF"])),
    ):
        for (label, color, marker), source in zip(styles, sources, strict=True):
            temperatures = np.asarray(sorted(source))
            means = np.asarray([source[t][0] for t in temperatures])
            sems = np.asarray([source[t][1] for t in temperatures])
            ax.errorbar(
                temperatures,
                means,
                yerr=sems,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                capsize=2.5,
                markerfacecolor="white" if marker == "^" else color,
                markeredgewidth=1.2,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"Mean nematic order, $\langle S\rangle$")
        ax.set_xticks(sorted(sources[0]))
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def end_result(force_field: str, run: str, rank: int, filename: str) -> dict[str, np.ndarray]:
    if rank == 1:
        from analyze_5cb_g1_director_paircorr import read_g1

        return read_g1(Path("data-processed/5cb/comparisons/g1-director-paircorr") / force_field.lower() / run / filename)
    root = Path("data-processed/5cb/amoeba-pt2.7/director-paircorr-end-to-end") if force_field == "AMOEBA" else Path("data-processed/5cb/gaff-zalign/director-paircorr-end-to-end")
    return read_result(root / run / filename)


def plot_pair_panels(
    core_rows: list[dict[str, str]],
    core_root: Path,
    report_root: Path,
    rank: int,
    perpendicular: bool,
    slab_halfwidth: float,
) -> None:
    quantity = f"g{rank}"
    amoeba = sorted(
        (row for row in core_rows if row["force_field"] == "AMOEBA"),
        key=lambda row: float(row["temperature_K"]),
    )
    gaff = sorted(
        (row for row in core_rows if row["force_field"] == "GAFF"),
        key=lambda row: float(row["temperature_K"]),
    )
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True)
    parallel_colors = {1.0: "#7b2cbf", 2.0: "#00a878", 3.0: "#0096c7"}
    cmap = plt.get_cmap("turbo", len(amoeba))
    alekos_parallel = read_csv_records(Path("data-processed/5cb/legacy/report-reproduction/alekos_data/alekos_page5_global_parallel.csv"))
    alekos_perp = read_csv_records(Path("data-processed/5cb/legacy/report-reproduction/alekos_data/alekos_page7_perpendicular.csv"))
    all_values: list[np.ndarray] = []

    for panel, row in enumerate(amoeba + gaff):
        ax = axes.flat[panel]
        force_field = row["force_field"]
        run = row["run"]
        temperature = int(float(row["temperature_K"]))
        color = cmap(panel) if force_field == "AMOEBA" else "#7b2cbf"
        filenames = ["perpendicular_slab.dat"] if perpendicular else [f"parallel_R{r:.1f}nm.dat" for r in RADII_NM]
        for file_index, filename in enumerate(filenames):
            radius = RADII_NM[file_index] if not perpendicular else None
            curve_color = color if perpendicular else parallel_colors[radius]
            end = end_result(force_field, run, rank, filename)
            core = read_core(core_root / force_field.lower() / run / filename)
            if perpendicular:
                end_x, end_y = end["x"], end[quantity]
                core_x, core_y = core["x"], core[quantity]
            else:
                end_x, end_y = mirror_parallel_curve(end["x"], end[quantity])
                core_x, core_y = mirror_parallel_curve(core["x"], core[quantity])
            ax.plot(end_x, end_y, color=curve_color, lw=1.2 if force_field == "AMOEBA" else 1.5, zorder=2)
            select = slice(None, None, 4)
            ax.scatter(
                core_x[select],
                core_y[select],
                s=10,
                marker="^",
                facecolors="none",
                edgecolors=curve_color,
                linewidths=0.55,
                alpha=0.9,
                zorder=4,
            )
            all_values.extend((end_y, core_y))

            if rank == 2:
                inertia = read_result(
                    Path("data-processed/5cb/comparisons/inertia-director-paircorr")
                    / force_field.lower()
                    / run
                    / filename
                )
                if perpendicular:
                    ix, iy = inertia["x"], inertia[quantity]
                else:
                    ix, iy = mirror_parallel_curve(inertia["x"], inertia[quantity])
                ax.scatter(ix[select], iy[select], s=5, marker="x", color=curve_color, linewidths=0.4, alpha=0.7, zorder=3)
                all_values.append(iy)

            if force_field == "GAFF":
                if rank == 1:
                    if perpendicular:
                        rx, ry = alekos_g1(alekos_perp, temperature, axis_reference="global_director")
                    else:
                        rx, ry = alekos_g1(alekos_parallel, temperature, radius=radius)
                else:
                    if perpendicular:
                        rx, ry = alekos_curve(alekos_perp, temperature, "g2", axis_reference="global_director")
                    else:
                        rx, ry = alekos_curve(alekos_parallel, temperature, "g2", radius=radius)
                if not perpendicular:
                    rx, ry = mirror_parallel_curve(rx, ry)
                ax.scatter(rx, ry, s=9, marker="o", color=curve_color, linewidths=0.35, alpha=0.78, zorder=5)
                all_values.append(ry)

        ax.axhline(0.0, color="0.45", ls=":", lw=0.7)
        suffix = "AMOEBA" if force_field == "AMOEBA" else "GAFF + Alekos"
        ax.set_title(f"{temperature} K · {suffix}", fontsize=9.8)
        ax.grid(alpha=0.16)

    axes.flat[-1].axis("off")
    handles: list[Line2D] = []
    if not perpendicular:
        handles.extend(
            Line2D([0], [0], color=parallel_colors[r], lw=2, label=f"R0={r:g} nm")
            for r in RADII_NM
        )
    handles.append(Line2D([0], [0], color="0.25", lw=1.4, label="End-to-end: line"))
    if rank == 2:
        handles.append(Line2D([0], [0], color="0.25", marker="x", markersize=4, lw=0, label="Inertia: ×"))
    handles.extend(
        (
            Line2D([0], [0], color="0.25", marker="^", markerfacecolor="none", markersize=5, lw=0, label="Alekos core C10→C04: △"),
            Line2D([0], [0], color="0.25", marker="o", markersize=4, lw=0, label="Alekos report: circles"),
        )
    )
    legend_title = f"Thin slab: |r·n(t)| < {slab_halfwidth:g} nm" if perpendicular else None
    axes.flat[-1].legend(handles=handles, title=legend_title, loc="center", frameon=False, fontsize=8.8)
    finite = np.concatenate([values[np.isfinite(values)] for values in all_values if values.size])
    if rank == 1:
        limit = 1.08 * max(abs(float(finite.min())), abs(float(finite.max())))
        axes.flat[0].set_ylim(-limit, limit)
    else:
        low = min(0.0, float(finite.min()))
        high = float(finite.max())
        margin = 0.06 * max(high - low, 1.0e-6)
        axes.flat[0].set_ylim(low - margin, high + margin)
    direction = "perpendicular to" if perpendicular else "parallel to"
    superscript = r"\perp" if perpendicular else r"\parallel"
    rank_name = "first-rank" if rank == 1 else "second-rank"
    for ax in axes[:, 0]:
        ax.set_ylabel(rf"$g_{rank}^{{{superscript}}}$")
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$r_\perp$ (nm)" if perpendicular else r"$z=\mathbf{r}_{ij}\cdot\mathbf{n}(t)$ (nm)")
    fig.suptitle(
        f"5CB {rank_name} correlations {direction} the director",
        fontsize=14,
    )
    report_root.mkdir(parents=True, exist_ok=True)
    name = f"{'perpendicular' if perpendicular else 'parallel'}_g{rank}_15panel_with_core_axis.png"
    fig.savefig(report_root / name, dpi=300)
    plt.close(fig)


def write_readme(output: Path, rows: list[dict[str, str | int | float]]) -> None:
    output.joinpath("README.md").write_text(
        "# 5CB Alekos-core-axis order and pair correlations\n\n"
        "The third molecular-axis definition is `u_core = unit(r_C04-r_C10)`, "
        "from local atom 10 (`C10`) to local atom 9 (`C04`). It is the `MolZax` "
        "definition in both supplied legacy Fortran versions. Here it is used "
        "consistently for the molecular P1/P2 weight and the framewise Q-tensor "
        "director. This differs from the supplied executable 11-Feb source, whose "
        "comment says MolZax but whose actual `Grobal_PAF` call uses the CN direction.\n\n"
        "All 11 AMOEBA and four GAFF trajectories contain 1000 frames of 1000 "
        "molecules. AMOEBA axes were produced by cpptraj after `autoimage`; each "
        "run retains its cpptraj input, log, and raw vector table. GAFF axes were "
        "formed with minimum-image C10-to-C04 displacement from the read-only GRO "
        "frames. Repository-local molecular COM/box caches were reused.\n\n"
        "For each axis definition separately, the same axes form Q, its largest-"
        "eigenvalue director, and the pair angular weight. Pair calculations use "
        "27 periodic images, R0=1,2,3 nm parallel cylinders, a 0.2 nm perpendicular "
        "slab, 200 direct bins from 0 to 5 nm, and no smoothing. Combined tables "
        "store raw counts and P1/P2 sums, g0, weighted g1/g2, and C1/C2.\n\n"
        f"Summary: `{output / 'summary.csv'}`\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amoeba-root", type=Path, default=AMOEBA_5CB.raw_path())
    parser.add_argument("--gaff-root", type=Path, default=GAFF_5CB.raw_path())
    parser.add_argument("--topology", type=Path, default=Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/topology/5cb-x1000-first-frame.pdb"))
    parser.add_argument("--amoeba-geometry-root", type=Path, default=Path("data-processed/5cb/amoeba-pt2.7/director-paircorr-end-to-end"))
    parser.add_argument("--gaff-geometry-root", type=Path, default=Path("data-processed/5cb/gaff-zalign/director-paircorr-end-to-end"))
    parser.add_argument("--out-data", type=Path, default=Path("data-processed/5cb/comparisons/alekos-core-axis"))
    parser.add_argument("--out-report", type=Path, default=Path("reports/5cb_alekos_core_axis"))
    parser.add_argument("--cpptraj", default="/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj")
    parser.add_argument("--nframes", type=int, default=1000)
    parser.add_argument("--maximum-nm", type=float, default=5.0)
    parser.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-axes", action="store_true")
    parser.add_argument("--force-correlations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.nframes <= 1000:
        raise SystemExit("--nframes must be between 1 and 1000")
    output = args.out_data.resolve()
    report = args.out_report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    amoeba_rows = read_csv(Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/summary.csv"))
    amoeba_rows.sort(key=lambda row: float(row["temperature_K"]))

    rows: list[dict[str, str | int | float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for row in amoeba_rows:
            run = row["run"]
            trajectory = args.amoeba_root / run / "5cb-x1000.arc"
            future = executor.submit(
                process_amoeba,
                run,
                float(row["temperature_K"]),
                str(trajectory.resolve()),
                str(args.topology.resolve()),
                str(args.amoeba_geometry_root.resolve()),
                str(output),
                args.cpptraj,
                args.nframes,
                args.maximum_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_axes,
                args.force_correlations,
            )
            futures[future] = f"AMOEBA/{run}"
        for run, temperature in GAFF_RUNS:
            future = executor.submit(
                process_gaff,
                run,
                temperature,
                str(args.gaff_root.resolve()),
                str(args.gaff_geometry_root.resolve()),
                str(output),
                args.nframes,
                args.maximum_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_axes,
                args.force_correlations,
            )
            futures[future] = f"GAFF/{run}"
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"Finished {futures[future]}", flush=True)

    rows.sort(key=lambda row: (str(row["force_field"]), float(row["temperature_K"])))
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_readme(output, rows)

    text_rows = [{key: str(value) for key, value in row.items()} for row in rows]
    plot_order(text_rows, report / "s_vs_temperature_three_axes.png")
    for rank in (1, 2):
        for perpendicular in (False, True):
            plot_pair_panels(
                text_rows,
                output,
                report,
                rank,
                perpendicular,
                args.slab_halfwidth_nm,
            )
    print(f"Wrote {output / 'summary.csv'}")
    print(f"Wrote figures beneath {report}")


if __name__ == "__main__":
    main()
