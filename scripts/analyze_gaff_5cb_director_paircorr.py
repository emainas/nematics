#!/usr/bin/env python3
"""Director-resolved 5CB pair correlations from the four GAFF GRO series.

The external GRO snapshots are read only.  Molecular orientation is the
minimum-image C18(local atom 19)->N01(local atom 4) end-to-end unit vector.
The framewise largest-eigenvalue eigenvector of its Q tensor is the director.
Generated geometry caches and correlation tables are written under data/.
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
    director_from_axes,
    write_table,
)
from nematics.gro import read_gro_frame  # noqa: E402


RUNS = (("gro", 356), ("gro2", 366), ("gro3", 376), ("gro4", 386))
NMOLECULES = 1000
NATOMS_PER_MOLECULE = 38
C18 = 18  # zero-based local atom 19
N01 = 3  # zero-based local atom 4
# Local atoms 1--19 comprise 18 C and one N (atom 4); atoms 20--38 are H.
MASSES = np.asarray(
    [14.007 if atom == N01 else (12.011 if atom < 19 else 1.008) for atom in range(38)]
)


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)\.gro", path.name)
    if match is None:
        raise ValueError(f"Not a numbered GRO frame: {path}")
    return int(match.group(1))


def frame_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = read_gro_frame(path, natoms_per_mol=NATOMS_PER_MOLECULE)
    if frame.positions.shape != (NMOLECULES, NATOMS_PER_MOLECULE, 3):
        raise ValueError(f"Unexpected molecular array in {path}: {frame.positions.shape}")

    # Reconstruct each molecule around C18 before computing its all-atom,
    # mass-weighted center. The resulting COM may lie outside the primary box;
    # the pair-search routine wraps it before generating periodic images.
    anchor = frame.positions[:, C18 : C18 + 1, :]
    displacement = frame.positions - anchor
    displacement -= frame.box[None, None, :] * np.rint(
        displacement / frame.box[None, None, :]
    )
    unwrapped = anchor + displacement
    centers = np.einsum("a,mac->mc", MASSES, unwrapped) / MASSES.sum()

    vectors = frame.positions[:, N01, :] - frame.positions[:, C18, :]
    vectors -= frame.box[None, :] * np.rint(vectors / frame.box[None, :])
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths == 0.0) or not np.isfinite(lengths).all():
        raise ValueError(f"Invalid C18-to-N01 vector in {path}")
    axes = vectors / lengths[:, None]
    return centers, frame.box, axes


def load_or_build_geometry(
    source: Path,
    output: Path,
    nframes: int | None,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    cache = output / "molecular_geometry.npz"
    if cache.is_file() and not force:
        stored = np.load(cache)
        return (
            stored["frame_numbers"],
            stored["centers_nm"],
            stored["boxes_nm"],
            stored["axes"],
            True,
        )

    paths = sorted(source.glob("frame*.gro"), key=frame_number)
    if nframes is not None:
        paths = paths[:nframes]
    if not paths:
        raise FileNotFoundError(f"No frame*.gro snapshots in {source}")

    centers = np.empty((len(paths), NMOLECULES, 3), dtype=float)
    boxes = np.empty((len(paths), 3), dtype=float)
    axes = np.empty_like(centers)
    numbers = np.empty(len(paths), dtype=int)
    for index, path in enumerate(paths):
        centers[index], boxes[index], axes[index] = frame_geometry(path)
        numbers[index] = frame_number(path)
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"  {source.name}: read frame {index + 1}/{len(paths)}", flush=True)
    np.savez(cache, frame_numbers=numbers, centers_nm=centers, boxes_nm=boxes, axes=axes)
    return numbers, centers, boxes, axes, False


def analyze_run(
    source_root: str,
    out_root: str,
    run_name: str,
    temperature: int,
    order_root: str,
    nframes: int | None,
    parallel_max_nm: float,
    perpendicular_max_nm: float,
    slab_halfwidth_nm: float,
    nbins: int,
    force_geometry: bool,
    force_analysis: bool,
) -> dict[str, str | int | float]:
    source = Path(source_root) / run_name
    output = Path(out_root) / run_name
    output.mkdir(parents=True, exist_ok=True)
    frame_numbers, centers, boxes, axes, reused_geometry = load_or_build_geometry(
        source, output, nframes, force_geometry
    )

    outputs = [output / f"parallel_R{radius:.1f}nm.dat" for radius in RADII_NM]
    outputs.append(output / "perpendicular_slab.dat")
    reused_analysis = all(path.is_file() for path in outputs) and not force_analysis
    if not reused_analysis:
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
        np.save(output / "S_recomputed_by_frame.npy", s_values)
    else:
        s_values = np.load(output / "S_recomputed_by_frame.npy")

    reference_path = Path(order_root) / f"{run_name}_end_to_end_S.npy"
    max_s_difference = float("nan")
    if reference_path.is_file():
        reference = np.load(reference_path)[: len(s_values)]
        max_s_difference = float(np.max(np.abs(reference - s_values)))

    # Also verify that the saved S follows directly from the cached axes.
    check_s = np.asarray([director_from_axes(frame_axes)[1] for frame_axes in axes])
    if not np.allclose(check_s, s_values, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"Stored S values disagree with cached axes for {run_name}")

    return {
        "run": run_name,
        "temperature_K": temperature,
        "temperature_mapping": "inferred_from_legacy_report_order",
        "source_gro_directory": str(source.resolve()),
        "frames": len(frame_numbers),
        "first_frame_number": int(frame_numbers[0]),
        "last_frame_number": int(frame_numbers[-1]),
        "molecules": NMOLECULES,
        "mean_S": float(s_values.mean()),
        "max_abs_S_difference_vs_saved_order": max_s_difference,
        "geometry_reused": str(reused_geometry),
        "correlations_reused": str(reused_analysis),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/hickory/proj/cdpoonlab/users/emainas/5cb-x1000-gaff-88x88x94-zalign"),
    )
    parser.add_argument(
        "--out-data", type=Path, default=Path("data/gaff_5cb_director_paircorr")
    )
    parser.add_argument(
        "--order-root", type=Path, default=Path("data/5cb_end_to_end_order")
    )
    parser.add_argument("--nframes", type=int)
    parser.add_argument("--parallel-max-nm", type=float, default=5.0)
    parser.add_argument("--perpendicular-max-nm", type=float, default=5.0)
    parser.add_argument("--slab-halfwidth-nm", type=float, default=0.2)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nframes is not None and args.nframes < 1:
        raise ValueError("--nframes must be positive")
    source_root = args.source_root.resolve()
    out_data = args.out_data.resolve()
    order_root = args.order_root.resolve()
    out_data.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_run,
                str(source_root),
                str(out_data),
                run_name,
                temperature,
                str(order_root),
                args.nframes,
                args.parallel_max_nm,
                args.perpendicular_max_nm,
                args.slab_halfwidth_nm,
                args.nbins,
                args.force_geometry,
                args.force_analysis,
            ): run_name
            for run_name, temperature in RUNS
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"Finished {row['run']} ({row['temperature_K']} K): "
                f"geometry_reused={row['geometry_reused']}, "
                f"analysis_reused={row['correlations_reused']}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["temperature_K"]))

    with (out_data / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out_data / "README.md").write_text(
        "# GAFF 5CB director-resolved pair correlations\n\n"
        "The four external `gro`, `gro2`, `gro3`, and `gro4` directories were read "
        "without modification. Their 356, 366, 376, and 386 K mapping is inferred "
        "from the legacy report ordering and is not encoded in the directory names.\n\n"
        "Each frame contains 1,000 5CB molecules of 38 atoms. The molecular axis is "
        "the minimum-image vector from local atom 19 (`C18`) to local atom 4 (`N01`). "
        "The framewise director is the largest-eigenvalue eigenvector of the Q tensor "
        "formed from these axes. Molecular centers are all-atom mass-weighted COMs; "
        "each molecule is first reconstructed about C18 with the orthorhombic "
        "minimum-image convention.\n\n"
        "The correlation definitions, 27 periodic images, 200 direct bins over "
        "0--5 nm, parallel cylinders R0=1,2,3 nm, and perpendicular slab "
        f"`|r dot n(t)| < {args.slab_halfwidth_nm:g} nm` match the AMOEBA comparison "
        "workflow. No smoothing is applied. Each table stores raw counts, raw P2 "
        "sums, normalized g0 and g2, and C2=g2/g0. The local NPZ geometry cache "
        "prevents rereading the external snapshots.\n"
    )
    print(f"Wrote {out_data / 'summary.csv'}")


if __name__ == "__main__":
    main()
