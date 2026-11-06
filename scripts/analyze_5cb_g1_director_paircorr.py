#!/usr/bin/env python3
"""First-rank director-resolved pair correlations for AMOEBA and GAFF 5CB.

The chemically directed molecular vector is C18(local atom 19)->N01(local
atom 4). For each frame its second-rank Q tensor defines the nematic director,
while pair weights use P1_ij = u_i dot u_j. The external trajectories remain
read only; existing repository-local centers, boxes, and axes are reused.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_amoeba_5cb_director_paircorr import (  # noqa: E402
    NMOLECULES,
    RADII_NM,
    candidate_pairs,
    director_basis,
    director_from_axes,
    load_cpptraj_axes,
    load_cpptraj_centers,
    read_csv_records,
    read_result,
)


GAFF_RUNS = (("gro", 356.0), ("gro2", 366.0), ("gro3", 376.0), ("gro4", 386.0))


def histogram_values(
    coordinate: np.ndarray,
    p1: np.ndarray,
    selected: np.ndarray,
    maximum: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = coordinate[selected]
    p1 = p1[selected]
    indices = np.floor(coordinate * nbins / maximum).astype(np.int64)
    valid = (indices >= 0) & (indices < nbins)
    return (
        np.bincount(indices[valid], minlength=nbins).astype(float),
        np.bincount(indices[valid], weights=p1[valid], minlength=nbins),
    )


def calculate_g1(
    centers_nm: np.ndarray,
    boxes_nm: np.ndarray,
    axes: np.ndarray,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    nframes = centers_nm.shape[0]
    parallel = {
        radius: {
            "count": np.zeros(nbins),
            "p1_sum": np.zeros(nbins),
            "g0_sum": np.zeros(nbins),
            "g1_sum": np.zeros(nbins),
        }
        for radius in RADII_NM
    }
    perpendicular = {
        "count": np.zeros(nbins),
        "p1_sum": np.zeros(nbins),
        "g0_sum": np.zeros(nbins),
        "g1_sum": np.zeros(nbins),
    }
    dz = parallel_max_nm / nbins
    r_edges = np.linspace(0.0, perpendicular_max_nm, nbins + 1)
    annular_factor = np.pi * slab_halfwidth_nm * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    for frame in range(nframes):
        director, _ = director_from_axes(axes[frame])
        basis = director_basis(director)
        pair_density = NMOLECULES * (NMOLECULES - 1.0) / float(np.prod(boxes_nm[frame]))

        first, second, displacement = candidate_pairs(
            centers_nm[frame],
            boxes_nm[frame],
            basis,
            np.asarray([max(RADII_NM), max(RADII_NM), parallel_max_nm]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        p1 = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        for radius in RADII_NM:
            selected = (z < parallel_max_nm) & (radial < radius)
            count, p1_sum = histogram_values(z, p1, selected, parallel_max_nm, nbins)
            norm = pair_density * np.pi * radius * radius * dz
            parallel[radius]["count"] += count
            parallel[radius]["p1_sum"] += p1_sum
            parallel[radius]["g0_sum"] += count / norm
            parallel[radius]["g1_sum"] += p1_sum / norm

        first, second, displacement = candidate_pairs(
            centers_nm[frame],
            boxes_nm[frame],
            basis,
            np.asarray([perpendicular_max_nm, perpendicular_max_nm, slab_halfwidth_nm]),
        )
        z = np.abs(displacement[:, 2])
        radial = np.hypot(displacement[:, 0], displacement[:, 1])
        p1 = np.einsum("ij,ij->i", axes[frame, first], axes[frame, second])
        selected = (z < slab_halfwidth_nm) & (radial < perpendicular_max_nm)
        count, p1_sum = histogram_values(
            radial, p1, selected, perpendicular_max_nm, nbins
        )
        norm = pair_density * annular_factor
        perpendicular["count"] += count
        perpendicular["p1_sum"] += p1_sum
        perpendicular["g0_sum"] += count / norm
        perpendicular["g1_sum"] += p1_sum / norm

        if (frame + 1) % 100 == 0 or frame + 1 == nframes:
            print(f"  processed g1 frame {frame + 1}/{nframes}", flush=True)

    for values in list(parallel.values()) + [perpendicular]:
        values["g0"] = values.pop("g0_sum") / nframes
        values["g1"] = values.pop("g1_sum") / nframes
        values["C1"] = np.divide(
            values["g1"],
            values["g0"],
            out=np.full(nbins, np.nan),
            where=values["g0"] > 0.0,
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
                values["g0"],
                values["g1"],
                values["C1"],
            )
        ),
        header="coordinate_nm raw_pair_count raw_P1_sum g0 g1_P1_weighted C1_equals_g1_over_g0",
        fmt=("%.8f", "%.0f", "%.10e", "%.10e", "%.10e", "%.10e"),
    )


def read_g1(path: Path) -> dict[str, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    return {
        "x": table[:, 0],
        "count": table[:, 1],
        "p1_sum": table[:, 2],
        "g0": table[:, 3],
        "g1": table[:, 4],
        "C1": table[:, 5],
    }


def max_g0_difference(output: Path, reference: Path) -> float:
    differences = []
    for radius in RADII_NM:
        name = f"parallel_R{radius:.1f}nm.dat"
        differences.append(
            np.nanmax(np.abs(read_g1(output / name)["g0"] - read_result(reference / name)["g0"]))
        )
    differences.append(
        np.nanmax(
            np.abs(
                read_g1(output / "perpendicular_slab.dat")["g0"]
                - read_result(reference / "perpendicular_slab.dat")["g0"]
            )
        )
    )
    return float(max(differences))


def write_run(
    output: Path,
    centers: np.ndarray,
    boxes: np.ndarray,
    axes: np.ndarray,
    reference: Path,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force: bool,
) -> tuple[bool, float]:
    expected = [output / f"parallel_R{radius:.1f}nm.dat" for radius in RADII_NM]
    expected.append(output / "perpendicular_slab.dat")
    reused = all(path.is_file() for path in expected) and not force
    if not reused:
        parallel, perpendicular = calculate_g1(
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
            write_table(output / f"parallel_R{radius:.1f}nm.dat", z, values)
        write_table(output / "perpendicular_slab.dat", r, perpendicular)
    return reused, max_g0_difference(output, reference)


def analyze_amoeba(
    run: str,
    temperature: float,
    geometry_root: str,
    vector_root: str,
    output_root: str,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / "amoeba" / run
    output.mkdir(parents=True, exist_ok=True)
    frames, centers, boxes = load_cpptraj_centers(
        Path(geometry_root) / run / "molecular_centers_and_box_cpptraj.dat"
    )
    vector_path = Path(vector_root) / run / "end_to_end_vectors.dat"
    vector_frames, axes = load_cpptraj_axes(vector_path)
    if not np.array_equal(frames, vector_frames):
        raise ValueError(f"Frame mismatch for {run}")
    reused, g0_error = write_run(
        output,
        centers,
        boxes,
        axes,
        Path(geometry_root) / run,
        parallel_max_nm,
        perpendicular_max_nm,
        slab_halfwidth_nm,
        nbins,
        force,
    )
    return {
        "force_field": "AMOEBA",
        "run": run,
        "temperature_K": temperature,
        "frames": len(frames),
        "molecules": NMOLECULES,
        "axis": "C18_to_N01",
        "max_abs_g0_difference_vs_g2_workflow": g0_error,
        "correlations_reused": str(reused),
    }


def analyze_gaff(
    run: str,
    temperature: float,
    geometry_root: str,
    output_root: str,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force: bool,
) -> dict[str, str | int | float]:
    stored = np.load(Path(geometry_root) / run / "molecular_geometry.npz")
    centers = stored["centers_nm"]
    boxes = stored["boxes_nm"]
    axes = stored["axes"]
    output = Path(output_root) / "gaff" / run
    output.mkdir(parents=True, exist_ok=True)
    reused, g0_error = write_run(
        output,
        centers,
        boxes,
        axes,
        Path(geometry_root) / run,
        parallel_max_nm,
        perpendicular_max_nm,
        slab_halfwidth_nm,
        nbins,
        force,
    )
    return {
        "force_field": "GAFF",
        "run": run,
        "temperature_K": temperature,
        "frames": centers.shape[0],
        "molecules": NMOLECULES,
        "axis": "C18_to_N01",
        "max_abs_g0_difference_vs_g2_workflow": g0_error,
        "correlations_reused": str(reused),
    }


def alekos_g1(
    records: list[dict[str, str]],
    temperature: int,
    radius: float | None = None,
    axis_reference: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for record in records:
        if int(record["temperature_K"]) != temperature or int(record["order"]) != 1:
            continue
        if radius is not None and int(record["radius_nm"]) != int(radius):
            continue
        if axis_reference is not None and record["axis_reference"] != axis_reference:
            continue
        coordinate = float(record["coordinate_nm"])
        if radius is not None and coordinate < 0.0:
            continue
        selected.append((coordinate, float(record["value"])))
    selected.sort()
    return (
        np.asarray([item[0] for item in selected]),
        np.asarray([item[1] for item in selected]),
    )


def mirror(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate((-x[::-1], x)), np.concatenate((y[::-1], y))


def plot_parallel(
    amoeba_rows: list[dict[str, str | int | float]],
    output_data: Path,
    alekos_root: Path,
    output_report: Path,
) -> None:
    colors = {1.0: "#7b2cbf", 2.0: "#00a878", 3.0: "#0096c7"}
    report = read_csv_records(alekos_root / "alekos_page5_global_parallel.csv")
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True)
    values_for_limits = []
    panel = 0
    for row in amoeba_rows:
        ax = axes.flat[panel]
        panel += 1
        for radius in RADII_NM:
            result = read_g1(
                output_data / "amoeba" / str(row["run"]) / f"parallel_R{radius:.1f}nm.dat"
            )
            x, y = mirror(result["x"], result["g1"])
            ax.plot(x, y, color=colors[radius], lw=1.1)
            values_for_limits.append(y)
        ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{float(row['temperature_K']):.0f} K · AMOEBA", fontsize=10)
        ax.grid(alpha=0.16)
    for run, temperature_float in GAFF_RUNS:
        temperature = int(temperature_float)
        ax = axes.flat[panel]
        panel += 1
        for radius in RADII_NM:
            result = read_g1(output_data / "gaff" / run / f"parallel_R{radius:.1f}nm.dat")
            x, y = mirror(result["x"], result["g1"])
            ax.plot(x, y, color=colors[radius], lw=1.5, zorder=2)
            values_for_limits.append(y)
            report_x, report_y = alekos_g1(report, temperature, radius=radius)
            report_x, report_y = mirror(report_x, report_y)
            ax.scatter(
                report_x,
                report_y,
                s=9,
                marker="o",
                color=colors[radius],
                linewidths=0.35,
                alpha=0.8,
                zorder=3,
            )
            values_for_limits.append(report_y)
        ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{temperature} K · GAFF + Alekos", fontsize=10)
        ax.grid(alpha=0.16)
    axes.flat[-1].axis("off")
    handles = [
        Line2D([0], [0], color=colors[r], lw=2, label=f"R0={r:g} nm") for r in RADII_NM
    ]
    handles.extend(
        [
            Line2D([0], [0], color="0.25", lw=1.5, label="Calculated: line"),
            Line2D([0], [0], color="0.25", marker="o", markersize=4, lw=0, label="Alekos: circles"),
        ]
    )
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False)
    finite = np.concatenate([v[np.isfinite(v)] for v in values_for_limits])
    limit = 1.08 * max(abs(float(finite.min())), abs(float(finite.max())))
    axes.flat[0].set_ylim(-limit, limit)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$g_1^{\parallel}(z;R_0)$")
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$z=\mathbf{r}_{ij}\cdot\mathbf{n}(t)$ (nm)")
    fig.suptitle("5CB first-rank correlations parallel to the director", fontsize=14)
    fig.savefig(output_report / "parallel_g1_15panel.png", dpi=300)
    plt.close(fig)


def plot_perpendicular(
    amoeba_rows: list[dict[str, str | int | float]],
    output_data: Path,
    alekos_root: Path,
    output_report: Path,
    slab_halfwidth_nm: float,
) -> None:
    cmap = plt.get_cmap("turbo", len(amoeba_rows))
    report = read_csv_records(alekos_root / "alekos_page7_perpendicular.csv")
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.8), sharex=True, sharey=True, constrained_layout=True)
    values_for_limits = []
    panel = 0
    for index, row in enumerate(amoeba_rows):
        ax = axes.flat[panel]
        panel += 1
        result = read_g1(output_data / "amoeba" / str(row["run"]) / "perpendicular_slab.dat")
        ax.plot(result["x"], result["g1"], color=cmap(index), lw=1.15)
        values_for_limits.append(result["g1"])
        ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{float(row['temperature_K']):.0f} K · AMOEBA", fontsize=10)
        ax.grid(alpha=0.16)
    for run, temperature_float in GAFF_RUNS:
        temperature = int(temperature_float)
        ax = axes.flat[panel]
        panel += 1
        result = read_g1(output_data / "gaff" / run / "perpendicular_slab.dat")
        ax.plot(result["x"], result["g1"], color="#7b2cbf", lw=1.5, zorder=2)
        values_for_limits.append(result["g1"])
        report_x, report_y = alekos_g1(
            report, temperature, axis_reference="global_director"
        )
        ax.scatter(
            report_x,
            report_y,
            s=9,
            marker="o",
            color="#7b2cbf",
            linewidths=0.35,
            alpha=0.8,
            zorder=3,
        )
        values_for_limits.append(report_y)
        ax.axhline(0.0, color="0.45", ls=":", lw=0.75)
        ax.set_title(f"{temperature} K · GAFF + Alekos", fontsize=10)
        ax.grid(alpha=0.16)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        handles=[
            Line2D([0], [0], color="0.25", lw=1.5, label="Calculated: line"),
            Line2D([0], [0], color="#7b2cbf", marker="o", markersize=4, lw=0, label="Alekos: circles"),
        ],
        title=f"|r·n(t)| < {slab_halfwidth_nm:g} nm",
        loc="center",
        frameon=False,
    )
    finite = np.concatenate([v[np.isfinite(v)] for v in values_for_limits])
    limit = 1.08 * max(abs(float(finite.min())), abs(float(finite.max())))
    axes.flat[0].set_ylim(-limit, limit)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$g_1^{\perp}(r_\perp)$")
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$r_\perp$ (nm)")
    fig.suptitle("5CB first-rank correlations perpendicular to the director", fontsize=14)
    fig.savefig(output_report / "perpendicular_g1_15panel.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amoeba-geometry-root", type=Path, default=Path("data/amoeba_5cb_director_paircorr")
    )
    parser.add_argument(
        "--amoeba-vector-root", type=Path, default=Path("data/amoeba_5cb_end_to_end_order")
    )
    parser.add_argument(
        "--gaff-geometry-root", type=Path, default=Path("data/gaff_5cb_director_paircorr")
    )
    parser.add_argument(
        "--alekos-root", type=Path, default=Path("data/5cb_report_reproduction/alekos_data")
    )
    parser.add_argument("--out-data", type=Path, default=Path("data/5cb_g1_director_paircorr"))
    parser.add_argument("--out-report", type=Path, default=Path("reports/5cb_g1_director_paircorr"))
    parser.add_argument("--parallel-max-nm", type=float, default=5.0)
    parser.add_argument("--perpendicular-max-nm", type=float, default=5.0)
    parser.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.amoeba_geometry_root / "summary.csv").open(newline="") as handle:
        amoeba_rows = list(csv.DictReader(handle))
    amoeba_rows.sort(key=lambda row: float(row["temperature_K"]))
    output_data = args.out_data.resolve()
    output_report = args.out_report.resolve()
    output_data.mkdir(parents=True, exist_ok=True)
    output_report.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for row in amoeba_rows:
            future = executor.submit(
                analyze_amoeba,
                row["run"],
                float(row["temperature_K"]),
                str(args.amoeba_geometry_root.resolve()),
                str(args.amoeba_vector_root.resolve()),
                str(output_data),
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force,
            )
            futures[future] = f"AMOEBA/{row['run']}"
        for run, temperature in GAFF_RUNS:
            future = executor.submit(
                analyze_gaff,
                run,
                temperature,
                str(args.gaff_geometry_root.resolve()),
                str(output_data),
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force,
            )
            futures[future] = f"GAFF/{run}"
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(f"Finished {futures[future]}", flush=True)

    rows.sort(key=lambda row: (str(row["force_field"]), float(row["temperature_K"])))
    with (output_data / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_parallel(amoeba_rows, output_data, args.alekos_root.resolve(), output_report)
    plot_perpendicular(
        amoeba_rows,
        output_data,
        args.alekos_root.resolve(),
        output_report,
        args.slab_halfwidth_nm,
    )
    (output_data / "README.md").write_text(
        "# 5CB first-rank director-resolved pair correlations\n\n"
        "The chemically directed molecular axis is `u_i = unit(r_N01-r_C18)`, "
        "from local atom 19 to local atom 4. Each frame's nematic director is "
        "the largest-eigenvalue eigenvector of the second-rank Q tensor built "
        "from these same axes. The first-rank pair weight is "
        "`P1_ij = u_i dot u_j`; therefore its sign distinguishes parallel and "
        "antiparallel molecular polarity.\n\n"
        "Parallel cylinders use R0=1,2,3 nm and `|r_ij dot n(t)|` from 0--5 nm. "
        f"Perpendicular correlations use `|r_ij dot n(t)| < {args.slab_halfwidth_nm:g} nm` "
        "and r_perp from 0--5 nm. There are 200 direct bins, no smoothing, and "
        "the same 27-image PBC and ideal-density normalization as the g2 workflow. "
        "Tables store raw count, raw P1 sum, g0, g1, and conditional C1=g1/g0.\n\n"
        "The plots show AMOEBA calculated lines in the first 11 panels. The final "
        "four panels overlay GAFF lines with the order-1 curves digitized from "
        "Alekos report pages 5 and 7 as circles. Only the report's global-director "
        "perpendicular curves are used. Parallel curves are mirrored from positive "
        "|z| for display. Inertia axes are intentionally excluded because their "
        "eigenvector signs are arbitrary and hence cannot define P1 without a "
        "separate chemical sign convention.\n"
    )
    print(f"Wrote {output_data / 'summary.csv'}")
    print(f"Wrote figures beneath {output_report}")


if __name__ == "__main__":
    main()
