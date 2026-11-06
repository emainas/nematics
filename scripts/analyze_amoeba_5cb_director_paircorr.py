#!/usr/bin/env python3
"""Framewise-director COM pair correlations for AMOEBA 5CB trajectories.

Cpptraj autoimages each trajectory and writes the mass-weighted molecular
centers and box lengths.  Previously retained cpptraj C18->N01 vectors supply
the molecular axes.  For every frame this program constructs the end-to-end Q
tensor, uses its largest-eigenvalue eigenvector as the director, and bins
positional and P2-weighted pair densities parallel and perpendicular to it.

The external ARC trajectories are read only.  All cpptraj inputs, logs, raw
geometry, numerical correlations, and plots are written in the repository.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


NMOLECULES = 1000
RADII_NM = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Run:
    label: str
    temperature: float
    trajectory: Path


def quote_cpptraj(path: Path) -> str:
    text = str(path)
    if '"' in text:
        raise ValueError(f"Double quotes are unsupported in cpptraj paths: {path}")
    return f'"{text}"'


def discover_runs(root: Path) -> list[Run]:
    runs: list[Run] = []
    for trajectory in sorted(root.glob("*/5cb-x1000.arc")):
        match = re.search(r"-(\d{3})$", trajectory.parent.name)
        if match is None:
            raise ValueError(f"Cannot infer temperature from {trajectory.parent.name}")
        runs.append(Run(trajectory.parent.name, float(match.group(1)), trajectory.resolve()))
    if len(runs) != 11:
        raise ValueError(f"Expected 11 AMOEBA 5CB trajectories, found {len(runs)}")
    return sorted(runs, key=lambda run: run.temperature)


def write_center_input(
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
    lines.extend(
        f"vector com_{residue:04d} center :{residue} out {quote_cpptraj(output)}"
        for residue in range(1, NMOLECULES + 1)
    )
    lines.extend(
        [
            f"vector box_lengths box out {quote_cpptraj(output)}",
            f"precision {quote_cpptraj(output)} 13 6",
            "run",
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


def load_cpptraj_centers(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    expected = 1 + 6 * (NMOLECULES + 1)
    if table.shape[1] != expected:
        raise ValueError(f"Expected {expected} columns in {path}, found {table.shape[1]}")
    frames = table[:, 0].astype(int)
    records = table[:, 1:].reshape(table.shape[0], NMOLECULES + 1, 6)
    centers_nm = records[:, :NMOLECULES, :3] / 10.0
    boxes_nm = records[:, NMOLECULES, :3] / 10.0
    if not np.isfinite(centers_nm).all() or not np.isfinite(boxes_nm).all():
        raise ValueError(f"Non-finite cpptraj center/box data in {path}")
    if np.any(boxes_nm <= 0.0):
        raise ValueError(f"Non-positive box length in {path}")
    return frames, centers_nm, boxes_nm


def load_cpptraj_axes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    frames = table[:, 0].astype(int)
    values = table[:, 1:]
    if values.shape[1] != 6 * NMOLECULES:
        raise ValueError(
            f"Expected {6 * NMOLECULES} vector columns in {path}, found {values.shape[1]}"
        )
    vectors = values.reshape(values.shape[0], NMOLECULES, 6)[:, :, :3]
    lengths = np.linalg.norm(vectors, axis=2)
    if np.any(lengths == 0.0) or not np.isfinite(lengths).all():
        raise ValueError(f"Invalid end-to-end vector in {path}")
    return frames, vectors / lengths[:, :, None]


def director_from_axes(axes: np.ndarray) -> tuple[np.ndarray, float]:
    q = 1.5 * np.einsum("mi,mj->ij", axes, axes) / axes.shape[0]
    q -= 0.5 * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    return eigenvectors[:, -1], float(eigenvalues[-1])


def director_basis(director: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(director @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(director, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(director, e1)
    return np.stack((e1, e2, director))


def candidate_pairs(
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    basis: np.ndarray,
    halfwidths_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return i<j periodic-image candidates inside an oriented bounding box."""
    wrapped = np.mod(positions_nm, box_nm)
    shifts = (
        np.asarray(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        )
        * box_nm
    )
    replicas = (wrapped[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    replica_ids = np.tile(np.arange(NMOLECULES), shifts.shape[0])
    original_rot = wrapped @ basis.T
    replica_rot = replicas @ basis.T
    scaled_original = original_rot / halfwidths_nm
    scaled_replicas = replica_rot / halfwidths_nm
    neighbors = cKDTree(scaled_replicas).query_ball_point(
        scaled_original, 1.0, p=np.inf, workers=1
    )
    lengths = np.fromiter((len(items) for items in neighbors), dtype=np.int64, count=NMOLECULES)
    if lengths.sum() == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty((0, 3), dtype=float)
    first = np.repeat(np.arange(NMOLECULES), lengths)
    replica_index = np.concatenate(neighbors)
    second = replica_ids[replica_index]
    keep = second > first
    first = first[keep]
    second = second[keep]
    displacement_rot = replica_rot[replica_index[keep]] - original_rot[first]
    return first, second, displacement_rot


def histogram_pair_values(
    coordinate: np.ndarray,
    p2: np.ndarray,
    selected: np.ndarray,
    maximum: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = coordinate[selected]
    p2 = p2[selected]
    indices = np.floor(coordinate * nbins / maximum).astype(np.int64)
    valid = (indices >= 0) & (indices < nbins)
    return (
        np.bincount(indices[valid], minlength=nbins).astype(float),
        np.bincount(indices[valid], weights=p2[valid], minlength=nbins),
    )


def calculate_correlations(
    centers_nm: np.ndarray,
    boxes_nm: np.ndarray,
    axes: np.ndarray,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray]:
    nframes = centers_nm.shape[0]
    parallel = {
        radius: {
            "count": np.zeros(nbins),
            "p2_sum": np.zeros(nbins),
            "g0_sum": np.zeros(nbins),
            "g2_sum": np.zeros(nbins),
        }
        for radius in RADII_NM
    }
    perpendicular = {
        "count": np.zeros(nbins),
        "p2_sum": np.zeros(nbins),
        "g0_sum": np.zeros(nbins),
        "g2_sum": np.zeros(nbins),
    }
    s_values = np.empty(nframes)
    dz = parallel_max_nm / nbins
    dr = perpendicular_max_nm / nbins
    r_edges = np.linspace(0.0, perpendicular_max_nm, nbins + 1)
    annular_factor = np.pi * slab_halfwidth_nm * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    for frame in range(nframes):
        director, s_values[frame] = director_from_axes(axes[frame])
        basis = director_basis(director)
        volume = float(np.prod(boxes_nm[frame]))
        pair_density = NMOLECULES * (NMOLECULES - 1.0) / volume

        first, second, displacement = candidate_pairs(
            centers_nm[frame],
            boxes_nm[frame],
            basis,
            np.asarray([max(RADII_NM), max(RADII_NM), parallel_max_nm]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        dot = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        p2 = 1.5 * dot * dot - 0.5
        for radius in RADII_NM:
            selected = (z < parallel_max_nm) & (radial < radius)
            count, p2_sum = histogram_pair_values(
                z, p2, selected, parallel_max_nm, nbins
            )
            norm = pair_density * np.pi * radius * radius * dz
            parallel[radius]["count"] += count
            parallel[radius]["p2_sum"] += p2_sum
            parallel[radius]["g0_sum"] += count / norm
            parallel[radius]["g2_sum"] += p2_sum / norm

        first, second, displacement = candidate_pairs(
            centers_nm[frame],
            boxes_nm[frame],
            basis,
            np.asarray([perpendicular_max_nm, perpendicular_max_nm, slab_halfwidth_nm]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        dot = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        p2 = 1.5 * dot * dot - 0.5
        selected = (z < slab_halfwidth_nm) & (radial < perpendicular_max_nm)
        count, p2_sum = histogram_pair_values(
            radial, p2, selected, perpendicular_max_nm, nbins
        )
        norm = pair_density * annular_factor
        perpendicular["count"] += count
        perpendicular["p2_sum"] += p2_sum
        perpendicular["g0_sum"] += count / norm
        perpendicular["g2_sum"] += p2_sum / norm

        if (frame + 1) % 100 == 0 or frame + 1 == nframes:
            print(f"  processed frame {frame + 1}/{nframes}", flush=True)

    for values in parallel.values():
        values["g0"] = values.pop("g0_sum") / nframes
        values["g2"] = values.pop("g2_sum") / nframes
        values["C2"] = np.divide(
            values["g2"], values["g0"], out=np.full(nbins, np.nan), where=values["g0"] > 0
        )
    perpendicular["g0"] = perpendicular.pop("g0_sum") / nframes
    perpendicular["g2"] = perpendicular.pop("g2_sum") / nframes
    perpendicular["C2"] = np.divide(
        perpendicular["g2"],
        perpendicular["g0"],
        out=np.full(nbins, np.nan),
        where=perpendicular["g0"] > 0,
    )
    return parallel, perpendicular, s_values


def write_table(path: Path, coordinate: np.ndarray, values: dict[str, np.ndarray]) -> None:
    matrix = np.column_stack(
        (
            coordinate,
            values["count"],
            values["p2_sum"],
            values["g0"],
            values["g2"],
            values["C2"],
        )
    )
    np.savetxt(
        path,
        matrix,
        header="coordinate_nm raw_pair_count raw_P2_sum g0 g2_P2_weighted C2_equals_g2_over_g0",
        fmt=("%.8f", "%.0f", "%.10e", "%.10e", "%.10e", "%.10e"),
    )


def analyze_run(
    run: Run,
    topology: str,
    vector_root: str,
    out_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force_cpptraj: bool,
    force_analysis: bool,
) -> dict[str, float | int | str]:
    out_dir = Path(out_root) / run.label
    out_dir.mkdir(parents=True, exist_ok=True)
    center_path = out_dir / "molecular_centers_and_box_cpptraj.dat"
    input_path = out_dir / "molecular_centers_and_box.cpptraj.in"
    log_path = out_dir / "molecular_centers_and_box.cpptraj.log"
    write_center_input(
        input_path, Path(topology), run.trajectory, center_path, start, stop, stride
    )
    reused_cpptraj = center_path.is_file() and not force_cpptraj
    if not reused_cpptraj:
        run_cpptraj(cpptraj, input_path, log_path)

    expected_outputs = [out_dir / f"parallel_R{radius:.1f}nm.dat" for radius in RADII_NM]
    expected_outputs.append(out_dir / "perpendicular_slab.dat")
    reused_analysis = all(path.is_file() for path in expected_outputs) and not force_analysis
    expected_frames = (stop - start) // stride + 1
    if not reused_analysis:
        center_frames, centers, boxes = load_cpptraj_centers(center_path)
        vector_path = Path(vector_root) / run.label / "end_to_end_vectors.dat"
        vector_frames, axes_all = load_cpptraj_axes(vector_path)
        wanted = np.arange(start, stop + 1, stride)
        vector_lookup = {frame: index for index, frame in enumerate(vector_frames)}
        try:
            vector_indices = np.asarray([vector_lookup[int(frame)] for frame in wanted])
        except KeyError as error:
            raise ValueError(f"Missing frame {error.args[0]} in {vector_path}") from error
        axes = axes_all[vector_indices]
        if not np.array_equal(center_frames, wanted):
            raise ValueError(
                f"Cpptraj center frames disagree for {run.label}: "
                f"{center_frames[:3]}... versus {wanted[:3]}..."
            )
        parallel, perpendicular, s_values = calculate_correlations(
            centers,
            boxes,
            axes,
            parallel_max_nm,
            perpendicular_max_nm,
            slab_halfwidth_nm,
            nbins,
        )
        z = (np.arange(nbins) + 0.5) * parallel_max_nm / nbins
        r = (np.arange(nbins) + 0.5) * perpendicular_max_nm / nbins
        for radius, values in parallel.items():
            write_table(out_dir / f"parallel_R{radius:.1f}nm.dat", z, values)
        write_table(out_dir / "perpendicular_slab.dat", r, perpendicular)
        np.save(out_dir / "S_recomputed_by_frame.npy", s_values)
        mean_s = float(s_values.mean())
    else:
        s_path = Path(vector_root) / run.label / "S_by_frame.npy"
        mean_s = float(np.load(s_path)[start - 1 : stop : stride].mean())

    return {
        "run": run.label,
        "temperature_K": run.temperature,
        "source_arc": str(run.trajectory),
        "frames": expected_frames,
        "molecules": NMOLECULES,
        "mean_S": mean_s,
        "cpptraj_geometry_reused": str(reused_cpptraj),
        "correlations_reused": str(reused_analysis),
    }


def read_result(path: Path) -> dict[str, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    return {
        "x": table[:, 0],
        "count": table[:, 1],
        "p2_sum": table[:, 2],
        "g0": table[:, 3],
        "g2": table[:, 4],
        "C2": table[:, 5],
    }


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def alekos_curve(
    records: list[dict[str, str]],
    temperature: int,
    quantity: str,
    radius: float | None = None,
    axis_reference: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def select_order(order: int) -> tuple[np.ndarray, np.ndarray]:
        selected = []
        for record in records:
            if int(record["temperature_K"]) != temperature or int(record["order"]) != order:
                continue
            if radius is not None and int(record["radius_nm"]) != int(radius):
                continue
            if axis_reference is not None and record["axis_reference"] != axis_reference:
                continue
            coordinate = float(record["coordinate_nm"])
            # The parallel report curves are symmetric on [-5, 5] whereas the
            # current calculation bins |z| on [0, 5]. Keep the matching half.
            if radius is not None and coordinate < 0.0:
                continue
            selected.append((coordinate, float(record["value"])))
        selected.sort()
        if not selected:
            return np.empty(0), np.empty(0)
        return np.asarray([item[0] for item in selected]), np.asarray([item[1] for item in selected])

    if quantity == "g0":
        return select_order(0)
    if quantity == "g2":
        return select_order(2)
    if quantity != "C2":
        raise ValueError(quantity)
    x0, g0 = select_order(0)
    x2, g2 = select_order(2)
    if x0.size == 0 or x2.size == 0:
        return np.empty(0), np.empty(0)
    inside = (x2 >= x0.min()) & (x2 <= x0.max())
    denominator = np.interp(x2, x0, g0)
    ratio = np.divide(g2, denominator, out=np.full_like(g2, np.nan), where=denominator > 0.05)
    physical = np.isfinite(ratio) & (ratio >= -0.55) & (ratio <= 1.05)
    keep = inside & physical
    return x2[keep], ratio[keep]


def set_quantity_limits(axes: np.ndarray, curves: list[np.ndarray], quantity: str) -> None:
    if quantity == "C2":
        axes.flat[0].set_ylim(-0.55, 1.05)
        return
    finite = np.concatenate([curve[np.isfinite(curve)] for curve in curves])
    if finite.size:
        low = min(0.0, float(finite.min()))
        high = float(finite.max())
        margin = 0.06 * max(high - low, 1.0e-6)
        axes.flat[0].set_ylim(low - margin, high + margin)


def mirror_parallel_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a curve binned at positive |z| onto signed negative and positive z."""
    return np.concatenate((-x[::-1], x)), np.concatenate((y[::-1], y))


def plot_parallel(
    rows: list[dict[str, float | int | str]],
    data_root: Path,
    gaff_data_root: Path,
    inertia_data_root: Path,
    report_root: Path,
    alekos_root: Path,
    quantity: str,
) -> None:
    labels = {
        "g0": r"$g_0^{\parallel}(z;R_0)$",
        "g2": r"$g_2^{\parallel}(z;R_0)$ (P$_2$ weighted)",
        "C2": r"$C_2^{\parallel}(z;R_0)=g_2/g_0$",
    }
    colors = {1.0: "#7b2cbf", 2.0: "#00a878", 3.0: "#0096c7"}
    alekos = read_csv_records(alekos_root / "alekos_page5_global_parallel.csv")
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True)
    curves: list[np.ndarray] = []
    panel = 0
    for row in rows:
        ax = axes.flat[panel]
        panel += 1
        for radius in RADII_NM:
            result = read_result(data_root / str(row["run"]) / f"parallel_R{radius:.1f}nm.dat")
            plot_x, plot_y = mirror_parallel_curve(result["x"], result[quantity])
            ax.plot(plot_x, plot_y, color=colors[radius], lw=1.1, label=f"R0={radius:g} nm")
            curves.append(result[quantity])
            inertia = read_result(
                inertia_data_root / "amoeba" / str(row["run"]) / f"parallel_R{radius:.1f}nm.dat"
            )
            inertia_x, inertia_y = mirror_parallel_curve(inertia["x"], inertia[quantity])
            ax.scatter(
                inertia_x,
                inertia_y,
                s=5,
                marker="x",
                color=colors[radius],
                linewidths=0.35,
                alpha=0.72,
                zorder=3,
            )
            curves.append(inertia[quantity])
        if quantity == "g0":
            ax.axhline(1.0, color="0.45", ls=":", lw=0.75)
        elif quantity in ("g2", "C2"):
            ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{float(row['temperature_K']):.0f} K", fontsize=10.5)
        ax.grid(alpha=0.16)
    gaff_runs = {356: "gro", 366: "gro2", 376: "gro3", 386: "gro4"}
    for temperature in (356, 366, 376, 386):
        ax = axes.flat[panel]
        panel += 1
        for radius in RADII_NM:
            result = read_result(
                gaff_data_root / gaff_runs[temperature] / f"parallel_R{radius:.1f}nm.dat"
            )
            plot_x, plot_y = mirror_parallel_curve(result["x"], result[quantity])
            ax.plot(
                plot_x, plot_y, color=colors[radius], lw=1.55, zorder=2
            )
            curves.append(result[quantity])
            inertia = read_result(
                inertia_data_root
                / "gaff"
                / gaff_runs[temperature]
                / f"parallel_R{radius:.1f}nm.dat"
            )
            inertia_x, inertia_y = mirror_parallel_curve(inertia["x"], inertia[quantity])
            ax.scatter(
                inertia_x,
                inertia_y,
                s=5,
                marker="x",
                color=colors[radius],
                linewidths=0.35,
                alpha=0.72,
                zorder=3,
            )
            curves.append(inertia[quantity])
            x, y = alekos_curve(alekos, temperature, quantity, radius=radius)
            plot_x, plot_y = mirror_parallel_curve(x, y)
            ax.scatter(
                plot_x,
                plot_y,
                s=9,
                marker="o",
                color=colors[radius],
                edgecolors=colors[radius],
                linewidths=0.45,
                alpha=0.78,
                zorder=4,
            )
            curves.append(y)
        if quantity == "g0":
            ax.axhline(1.0, color="0.45", ls=":", lw=0.75)
        else:
            ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{temperature} K · GAFF + Alekos", fontsize=9.7)
        ax.grid(alpha=0.16)
    axes.flat[-1].axis("off")
    radius_handles = [
        Line2D([0], [0], color=colors[radius], lw=2, label=f"R0={radius:g} nm")
        for radius in RADII_NM
    ]
    source_handles = [
        Line2D([0], [0], color="0.25", lw=1.2, label="End-to-end: line"),
        Line2D(
            [0], [0], color="0.25", marker="x", markersize=4, lw=0,
            markeredgewidth=0.7, label="Inertia axis: small ×",
        ),
        Line2D(
            [0], [0], color="0.25", marker="o", markersize=4, lw=0,
            label="Alekos report: circles",
        ),
    ]
    axes.flat[-1].legend(
        handles=radius_handles + source_handles,
        loc="center",
        frameon=False,
    )
    set_quantity_limits(axes, curves, quantity)
    for ax in axes[:, 0]:
        ax.set_ylabel(labels[quantity])
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$z=\mathbf{r}_{ij}\cdot\mathbf{n}(t)$ (nm)")
    fig.suptitle(
        "5CB parallel correlations: end-to-end lines vs inertia-axis markers",
        fontsize=14,
    )
    fig.savefig(report_root / f"parallel_{quantity}_11panel.png", dpi=300)
    plt.close(fig)


def plot_perpendicular(
    rows: list[dict[str, float | int | str]],
    data_root: Path,
    gaff_data_root: Path,
    inertia_data_root: Path,
    report_root: Path,
    alekos_root: Path,
    quantity: str,
    slab_halfwidth_nm: float,
) -> None:
    labels = {
        "g0": r"$g_0^{\perp}(r_\perp)$",
        "g2": r"$g_2^{\perp}(r_\perp)$ (P$_2$ weighted)",
        "C2": r"$C_2^{\perp}(r_\perp)=g_2/g_0$",
    }
    cmap = plt.get_cmap("turbo", len(rows))
    alekos = read_csv_records(alekos_root / "alekos_page7_perpendicular.csv")
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True)
    curves: list[np.ndarray] = []
    panel = 0
    for index, row in enumerate(rows):
        ax = axes.flat[panel]
        panel += 1
        result = read_result(data_root / str(row["run"]) / "perpendicular_slab.dat")
        ax.plot(result["x"], result[quantity], color=cmap(index), lw=1.15)
        curves.append(result[quantity])
        inertia = read_result(
            inertia_data_root / "amoeba" / str(row["run"]) / "perpendicular_slab.dat"
        )
        ax.scatter(
            inertia["x"],
            inertia[quantity],
            s=5,
            marker="x",
            color=cmap(index),
            linewidths=0.35,
            alpha=0.72,
            zorder=3,
        )
        curves.append(inertia[quantity])
        if quantity == "g0":
            ax.axhline(1.0, color="0.45", ls=":", lw=0.75)
        elif quantity in ("g2", "C2"):
            ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{float(row['temperature_K']):.0f} K", fontsize=10.5)
        ax.grid(alpha=0.16)
    gaff_runs = {356: "gro", 366: "gro2", 376: "gro3", 386: "gro4"}
    for temperature in (356, 366, 376, 386):
        ax = axes.flat[panel]
        panel += 1
        result = read_result(
            gaff_data_root / gaff_runs[temperature] / "perpendicular_slab.dat"
        )
        ax.plot(result["x"], result[quantity], color="#7b2cbf", lw=1.55, zorder=2)
        curves.append(result[quantity])
        inertia = read_result(
            inertia_data_root
            / "gaff"
            / gaff_runs[temperature]
            / "perpendicular_slab.dat"
        )
        ax.scatter(
            inertia["x"],
            inertia[quantity],
            s=5,
            marker="x",
            color="#7b2cbf",
            linewidths=0.35,
            alpha=0.72,
            zorder=3,
        )
        curves.append(inertia[quantity])
        x, y = alekos_curve(
            alekos,
            temperature,
            quantity,
            axis_reference="global_director",
        )
        ax.scatter(
            x,
            y,
            s=9,
            marker="o",
            color="#7b2cbf",
            edgecolors="#7b2cbf",
            linewidths=0.45,
            alpha=0.78,
            zorder=4,
        )
        curves.append(y)
        if quantity == "g0":
            ax.axhline(1.0, color="0.45", ls=":", lw=0.75)
        else:
            ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{temperature} K · GAFF + Alekos", fontsize=9.7)
        ax.grid(alpha=0.16)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        handles=[
            Line2D([0], [0], color="0.25", lw=1.2, label="End-to-end: line"),
            Line2D(
                [0], [0], color="0.25", marker="x", markersize=4, lw=0,
                markeredgewidth=0.7, label="Inertia axis: small ×",
            ),
            Line2D(
                [0], [0], color="#7b2cbf", marker="o", markersize=4, lw=0,
                label="Alekos report: circles",
            ),
        ],
        title=f"Thin slab: |r·n(t)| < {slab_halfwidth_nm:g} nm",
        loc="center",
        frameon=False,
    )
    set_quantity_limits(axes, curves, quantity)
    for ax in axes[:, 0]:
        ax.set_ylabel(labels[quantity])
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$r_\perp$ (nm)")
    fig.suptitle(
        "5CB perpendicular correlations: end-to-end lines vs inertia-axis markers",
        fontsize=14,
    )
    fig.savefig(report_root / f"perpendicular_{quantity}_11panel.png", dpi=300)
    plt.close(fig)


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/hickory/proj/cdpoonlab/users/emainas/tinker9-5cb-x1000-pt2.7"),
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data/amoeba_5cb_end_to_end_order/topology/5cb-x1000-first-frame.pdb"),
    )
    parser.add_argument(
        "--vector-root", type=Path, default=Path("data/amoeba_5cb_end_to_end_order")
    )
    parser.add_argument(
        "--out-data", type=Path, default=Path("data/amoeba_5cb_director_paircorr")
    )
    parser.add_argument(
        "--out-report", type=Path, default=Path("reports/amoeba_5cb_director_paircorr")
    )
    parser.add_argument(
        "--alekos-data-root",
        type=Path,
        default=Path("data/5cb_report_reproduction/alekos_data"),
    )
    parser.add_argument(
        "--gaff-data-root",
        type=Path,
        default=Path("data/gaff_5cb_director_paircorr"),
    )
    parser.add_argument(
        "--inertia-data-root",
        type=Path,
        default=Path("data/5cb_inertia_director_paircorr"),
    )
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--parallel-max-nm", type=float, default=5.0)
    parser.add_argument("--perpendicular-max-nm", type=float, default=5.0)
    parser.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-cpptraj", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop < args.start or args.stride < 1:
        raise ValueError("Invalid frame range")
    runs = discover_runs(args.data_root.resolve())
    topology = args.topology.resolve()
    vector_root = args.vector_root.resolve()
    if not topology.is_file():
        raise FileNotFoundError(topology)
    cpptraj = shutil.which(args.cpptraj)
    if cpptraj is None:
        raise SystemExit(f"cpptraj executable not found: {args.cpptraj}")
    out_data = args.out_data.resolve()
    out_report = args.out_report.resolve()
    alekos_root = args.alekos_data_root.resolve()
    gaff_data_root = args.gaff_data_root.resolve()
    inertia_data_root = args.inertia_data_root.resolve()
    if not (alekos_root / "alekos_page5_global_parallel.csv").is_file():
        raise FileNotFoundError(alekos_root / "alekos_page5_global_parallel.csv")
    if not (alekos_root / "alekos_page7_perpendicular.csv").is_file():
        raise FileNotFoundError(alekos_root / "alekos_page7_perpendicular.csv")
    for run_name in ("gro", "gro2", "gro3", "gro4"):
        if not (gaff_data_root / run_name / "perpendicular_slab.dat").is_file():
            raise FileNotFoundError(gaff_data_root / run_name / "perpendicular_slab.dat")
        if not (inertia_data_root / "gaff" / run_name / "perpendicular_slab.dat").is_file():
            raise FileNotFoundError(
                inertia_data_root / "gaff" / run_name / "perpendicular_slab.dat"
            )
    out_data.mkdir(parents=True, exist_ok=True)
    out_report.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_run,
                run,
                str(topology),
                str(vector_root),
                str(out_data),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_cpptraj,
                args.force_analysis,
            ): run
            for run in runs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"Finished {row['run']} ({float(row['temperature_K']):.0f} K): "
                f"cpptraj_reused={row['cpptraj_geometry_reused']}, "
                f"analysis_reused={row['correlations_reused']}",
                flush=True,
            )

    rows.sort(key=lambda row: float(row["temperature_K"]))
    write_summary(rows, out_data / "summary.csv")
    for quantity in ("g0", "g2", "C2"):
        plot_parallel(
            rows,
            out_data,
            gaff_data_root,
            inertia_data_root,
            out_report,
            alekos_root,
            quantity,
        )
        plot_perpendicular(
            rows,
            out_data,
            gaff_data_root,
            inertia_data_root,
            out_report,
            alekos_root,
            quantity,
            args.slab_halfwidth_nm,
        )

    (out_data / "README.md").write_text(
        "# AMOEBA 5CB director-resolved pair correlations\n\n"
        "Cpptraj 7.6.2 read the external ARC trajectories without modifying them. "
        "It used `autoimage`, then wrote the mass-weighted COM of every 5CB residue "
        "and the frame box lengths. The existing cpptraj C18-to-N01 vector tables "
        "supply the molecular end-to-end axes. For every frame, the largest-eigenvalue "
        "eigenvector of the end-to-end Q tensor defines `n(t)`.\n\n"
        "For each periodic molecular pair image, `z = |r_ij dot n(t)|`, "
        "`r_perp = sqrt(r_ij^2-z^2)`, and "
        "`P2 = [3(u_i dot u_j)^2-1]/2`. Parallel histograms use cylinders with "
        "R0 = 1, 2, and 3 nm. Perpendicular histograms use the legacy thin-slab "
        f"selection `|r_ij dot n(t)| < {args.slab_halfwidth_nm:g} nm`. The range is "
        f"0-{args.parallel_max_nm:g} nm with {args.nbins} direct bins and no smoothing.\n\n"
        "Each table contains raw pair count, raw P2 sum, ideal-density-normalized "
        "positional `g0`, ideal-density-normalized P2-weighted `g2`, and the conditional "
        "orientation `C2 = g2/g0`. Periodic replicas are included, following the legacy "
        "27-image convention; R0=3 nm is smaller than half the shortest box length. "
        "Every run directory retains the exact cpptraj input, log, raw DAT geometry, "
        "and final DAT correlation tables.\n"
        "The parallel DAT tables retain the directly accumulated `|z|` bins on "
        "0--5 nm. For display only, the comparison figures mirror those values "
        "onto negative z using the nematic symmetry `g(+z)=g(-z)`; no negative-z "
        "samples are recalculated.\n"
        "\nThe six comparison figures also include the digitized Alekos report "
        "curves from `data/5cb_report_reproduction/alekos_data/`. The final four "
        "panels overlay solid lines computed from the GAFF GRO series (`gro`, "
        "`gro2`, `gro3`, and `gro4`) at the inferred temperatures 356, 366, 376, "
        "and 386 K with the report data drawn as thick circles. Report `g0` and "
        "`g2` are direct digitizations; report "
        "`C2` markers are derived by interpolating the digitized `g0` onto the "
        "digitized `g2` coordinates and taking `g2/g0`. These markers are read "
        "directly from the report, which describes Z as the macroscopic director "
        "and molecular z as the long molecular axis; no CN-based director is "
        "introduced in this comparison. The AMOEBA curves specifically use the "
        "C18-to-N01 end-to-end definition for both Q and P2. The GAFF lines use "
        "the same axis and framewise-Q definition; their numerical tables and "
        "provenance are in `data/gaff_5cb_director_paircorr/`. Small x markers "
        "show the corresponding calculation using each molecule's smallest-moment "
        "inertia eigenvector consistently for both the framewise Q director and "
        "pair P2. Those tables are in `data/5cb_inertia_director_paircorr/`.\n"
    )
    print(f"Wrote {out_data / 'summary.csv'}")
    print(f"Wrote plots beneath {out_report}")


if __name__ == "__main__":
    main()
