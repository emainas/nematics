#!/usr/bin/env python3
"""5CB pair correlations using the smallest-moment molecular inertia axis.

For every frame, the molecular principal axes are used both to construct the
Q-tensor director and in P2(u_i dot u_j). Existing cpptraj inertia axes are
reused for AMOEBA. GAFF inertia axes are calculated from the read-only GRO
snapshots and cached in the repository.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_amoeba_5cb_director_paircorr import (  # noqa: E402
    RADII_NM,
    calculate_correlations,
    load_cpptraj_centers,
    write_table,
)
from nematics.gro import read_gro_frame  # noqa: E402


NMOLECULES = 1000
NATOMS_PER_MOLECULE = 38
C18 = 18  # zero-based local atom 19
MASSES = np.asarray(
    [14.007 if atom == 3 else (12.011 if atom < 19 else 1.008) for atom in range(38)]
)
GAFF_RUNS = (("gro", 356), ("gro2", 366), ("gro3", 376), ("gro4", 386))


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)\.gro", path.name)
    if match is None:
        raise ValueError(f"Not a numbered GRO frame: {path}")
    return int(match.group(1))


def write_correlations(
    output: Path,
    centers: np.ndarray,
    boxes: np.ndarray,
    axes: np.ndarray,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force: bool,
) -> tuple[np.ndarray, bool]:
    expected = [output / f"parallel_R{radius:.1f}nm.dat" for radius in RADII_NM]
    expected.append(output / "perpendicular_slab.dat")
    s_path = output / "S_by_frame.npy"
    reused = all(path.is_file() for path in expected) and s_path.is_file() and not force
    if reused:
        return np.load(s_path), True

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
        write_table(output / f"parallel_R{radius:.1f}nm.dat", z, values)
    write_table(output / "perpendicular_slab.dat", r, perpendicular)
    np.save(s_path, s_values)
    return s_values, False


def analyze_amoeba(
    run: str,
    temperature: float,
    end_pair_root: str,
    inertia_order_root: str,
    output_root: str,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / "amoeba" / run
    output.mkdir(parents=True, exist_ok=True)
    _, centers, boxes = load_cpptraj_centers(
        Path(end_pair_root) / run / "molecular_centers_and_box_cpptraj.dat"
    )
    axis_path = Path(inertia_order_root) / run / "principal_z_vectors.npy"
    axes = np.load(axis_path)
    axes /= np.linalg.norm(axes, axis=2)[:, :, None]
    if centers.shape != axes.shape:
        raise ValueError(f"AMOEBA center/axis shape mismatch for {run}")

    s_values, reused = write_correlations(
        output,
        centers,
        boxes,
        axes,
        parallel_max_nm,
        perpendicular_max_nm,
        slab_halfwidth_nm,
        nbins,
        force,
    )
    reference = np.load(Path(inertia_order_root) / run / "S_by_frame.npy")
    max_s_difference = float(np.max(np.abs(s_values - reference)))
    return {
        "force_field": "AMOEBA",
        "run": run,
        "temperature_K": temperature,
        "frames": axes.shape[0],
        "molecules": axes.shape[1],
        "mean_S_inertia": float(s_values.mean()),
        "max_abs_S_difference_vs_inertia_order": max_s_difference,
        "max_COM_difference_nm": 0.0,
        "axes_source": str(axis_path.resolve()),
        "correlations_reused": str(reused),
    }


def inertia_axes_from_gro(
    source: Path,
    wanted_frames: np.ndarray,
    cached_centers: np.ndarray,
    boxes: np.ndarray,
    output: Path,
    force: bool,
) -> tuple[np.ndarray, float, bool]:
    axis_path = output / "inertia_smallest_moment_axes.npy"
    if axis_path.is_file() and not force:
        return np.load(axis_path), float("nan"), True

    paths_by_number = {
        frame_number(path): path for path in source.glob("frame*.gro")
    }
    axes = np.empty_like(cached_centers)
    max_com_difference = 0.0
    for index, number in enumerate(wanted_frames):
        try:
            path = paths_by_number[int(number)]
        except KeyError as error:
            raise FileNotFoundError(f"Missing frame{number}.gro in {source}") from error
        frame = read_gro_frame(path, natoms_per_mol=NATOMS_PER_MOLECULE)
        anchor = frame.positions[:, C18 : C18 + 1, :]
        displacement = frame.positions - anchor
        displacement -= frame.box[None, None, :] * np.rint(
            displacement / frame.box[None, None, :]
        )
        unwrapped = anchor + displacement
        centers = np.einsum("a,mac->mc", MASSES, unwrapped) / MASSES.sum()
        centered = unwrapped - centers[:, None, :]

        outer = np.einsum("a,mai,maj->mij", MASSES, centered, centered)
        radius_squared = np.einsum("a,mai,mai->m", MASSES, centered, centered)
        inertia = -outer
        diagonal = np.arange(3)
        inertia[:, diagonal, diagonal] += radius_squared[:, None]
        _, eigenvectors = np.linalg.eigh(inertia)
        axes[index] = eigenvectors[:, :, 0]

        com_delta = centers - cached_centers[index]
        com_delta -= boxes[index][None, :] * np.rint(com_delta / boxes[index][None, :])
        max_com_difference = max(
            max_com_difference, float(np.max(np.linalg.norm(com_delta, axis=1)))
        )
        if (index + 1) % 100 == 0 or index + 1 == len(wanted_frames):
            print(f"  {source.name}: inertia frame {index + 1}/{len(wanted_frames)}", flush=True)
    np.save(axis_path, axes)
    return axes, max_com_difference, False


def analyze_gaff(
    run: str,
    temperature: float,
    source_root: str,
    end_pair_root: str,
    output_root: str,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force_axes: bool,
    force_correlations: bool,
) -> dict[str, str | int | float]:
    output = Path(output_root) / "gaff" / run
    output.mkdir(parents=True, exist_ok=True)
    geometry = np.load(Path(end_pair_root) / run / "molecular_geometry.npz")
    frame_numbers = geometry["frame_numbers"]
    centers = geometry["centers_nm"]
    boxes = geometry["boxes_nm"]
    axes, max_com_difference, reused_axes = inertia_axes_from_gro(
        Path(source_root) / run,
        frame_numbers,
        centers,
        boxes,
        output,
        force_axes,
    )
    axes /= np.linalg.norm(axes, axis=2)[:, :, None]
    s_values, reused_correlations = write_correlations(
        output,
        centers,
        boxes,
        axes,
        parallel_max_nm,
        perpendicular_max_nm,
        slab_halfwidth_nm,
        nbins,
        force_correlations,
    )
    return {
        "force_field": "GAFF",
        "run": run,
        "temperature_K": temperature,
        "frames": axes.shape[0],
        "molecules": axes.shape[1],
        "mean_S_inertia": float(s_values.mean()),
        "max_abs_S_difference_vs_inertia_order": float("nan"),
        "max_COM_difference_nm": max_com_difference,
        "axes_source": str((output / "inertia_smallest_moment_axes.npy").resolve()),
        "correlations_reused": str(reused_correlations),
        "axes_reused": str(reused_axes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amoeba-end-pair-root",
        type=Path,
        default=Path("data/amoeba_5cb_director_paircorr"),
    )
    parser.add_argument(
        "--amoeba-inertia-order-root",
        type=Path,
        default=Path("data/amoeba_5cb_inertia_order"),
    )
    parser.add_argument(
        "--gaff-source-root",
        type=Path,
        default=Path("/hickory/proj/cdpoonlab/users/emainas/5cb-x1000-gaff-88x88x94-zalign"),
    )
    parser.add_argument(
        "--gaff-end-pair-root",
        type=Path,
        default=Path("data/gaff_5cb_director_paircorr"),
    )
    parser.add_argument(
        "--out-data", type=Path, default=Path("data/5cb_inertia_director_paircorr")
    )
    parser.add_argument("--parallel-max-nm", type=float, default=5.0)
    parser.add_argument("--perpendicular-max-nm", type=float, default=5.0)
    parser.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-gaff-axes", action="store_true")
    parser.add_argument("--force-correlations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amoeba_summary = Path(args.amoeba_inertia_order_root) / "summary.csv"
    with amoeba_summary.open(newline="") as handle:
        amoeba_rows = list(csv.DictReader(handle))
    out_data = args.out_data.resolve()
    out_data.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for row in amoeba_rows:
            future = executor.submit(
                analyze_amoeba,
                row["run"],
                float(row["temperature_K"]),
                str(args.amoeba_end_pair_root.resolve()),
                str(args.amoeba_inertia_order_root.resolve()),
                str(out_data),
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_correlations,
            )
            futures[future] = f"AMOEBA/{row['run']}"
        for run, temperature in GAFF_RUNS:
            future = executor.submit(
                analyze_gaff,
                run,
                temperature,
                str(args.gaff_source_root.resolve()),
                str(args.gaff_end_pair_root.resolve()),
                str(out_data),
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_gaff_axes,
                args.force_correlations,
            )
            futures[future] = f"GAFF/{run}"
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(f"Finished {futures[future]}", flush=True)

    rows.sort(key=lambda row: (str(row["force_field"]), float(row["temperature_K"])))
    fieldnames = sorted({key for row in rows for key in row})
    with (out_data / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_data / "README.md").write_text(
        "# 5CB inertia-axis director-resolved pair correlations\n\n"
        "For every molecule, `u_i` is the eigenvector of the mass-weighted "
        "molecular inertia tensor having the smallest principal moment. The "
        "framewise Q tensor and its director are constructed from these same "
        "inertia axes. Pair orientation is `P2=[3(u_i dot u_j)^2-1]/2`. Thus "
        "the molecular axis, director, and pair weight use one consistent "
        "definition.\n\n"
        "AMOEBA reuses the cpptraj principal-z vectors under "
        "`data/amoeba_5cb_inertia_order/` and the previously saved cpptraj COMs "
        "and boxes. GAFF axes are diagonalized directly from mass-weighted "
        "inertia tensors after minimum-image molecular reconstruction from each "
        "read-only GRO snapshot, then saved here. GAFF COMs and boxes are reused "
        "from `data/gaff_5cb_director_paircorr/`.\n\n"
        "The binning, normalization, PBC images, parallel R0=1,2,3 nm cylinders, "
        f"and perpendicular `|r dot n(t)| < {args.slab_halfwidth_nm:g} nm` slab "
        "match the end-to-end workflow. No KDE or smoothing is used.\n"
    )
    print(f"Wrote {out_data / 'summary.csv'}")


if __name__ == "__main__":
    main()
