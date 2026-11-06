#!/usr/bin/env python3
"""Calculate and plot molecular COM-COM RDFs for AMOEBA 5CB trajectories.

Cpptraj reconstructs molecules with autoimage and calculates a spherical RDF
between mass-weighted molecular centers. Raw cpptraj inputs, logs, normalized
RDFs, and pair-count histograms are retained and reused by default.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nematics.datasets import get_dataset  # noqa: E402


AMOEBA_5CB = get_dataset("amoeba-5cb")


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
    return [
        Run(run.id, run.temperature_k, trajectory.resolve())
        for run, trajectory in AMOEBA_5CB.run_trajectories(root)
    ]


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    rdf_path: Path,
    raw_path: Path,
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
                f"trajin {quote_cpptraj(trajectory)} {start} {stop} {stride}",
                "autoimage",
                (
                    f"radial out {quote_cpptraj(rdf_path)} {spacing_a:g} "
                    f"{maximum_a:g} :1-1000 :1-1000 bymol1 bymol2 mass "
                    f"volume rawrdf {quote_cpptraj(raw_path)} com_com_rdf"
                ),
                f"precision {quote_cpptraj(rdf_path)} 14 8",
                f"precision {quote_cpptraj(raw_path)} 18 6",
                "run",
                "quit",
                "",
            ]
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


def analyze_run(
    run: Run,
    topology: str,
    out_root: str,
    cpptraj: str,
    start: int,
    stop: int,
    stride: int,
    spacing_a: float,
    maximum_a: float,
    force_cpptraj: bool,
) -> dict[str, float | int | str]:
    out_dir = Path(out_root) / run.label
    out_dir.mkdir(parents=True, exist_ok=True)
    rdf_path = out_dir / "com_com_rdf_cpptraj.dat"
    raw_path = out_dir / "com_com_rdf_raw_counts.dat"
    input_path = out_dir / "com_com_rdf.cpptraj.in"
    log_path = out_dir / "com_com_rdf.cpptraj.log"
    write_cpptraj_input(
        input_path,
        Path(topology),
        run.trajectory,
        rdf_path,
        raw_path,
        start,
        stop,
        stride,
        spacing_a,
        maximum_a,
    )
    reused = rdf_path.is_file() and raw_path.is_file() and not force_cpptraj
    if not reused:
        run_cpptraj(cpptraj, input_path, log_path)

    rdf = np.loadtxt(rdf_path, comments="#", ndmin=2)
    raw = np.loadtxt(raw_path, comments="#", ndmin=2)
    if rdf.shape != raw.shape or rdf.shape[1] != 2:
        raise ValueError(f"Unexpected cpptraj RDF shapes for {run.label}: {rdf.shape}, {raw.shape}")
    if not np.allclose(rdf[:, 0], raw[:, 0]):
        raise ValueError(f"RDF and raw-count grids disagree for {run.label}")
    if not np.isfinite(rdf).all() or np.any(rdf[:, 1] < 0.0):
        raise ValueError(f"Invalid RDF values for {run.label}")

    csv_path = out_dir / "com_com_rdf.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("r_nm", "g_com_com", "raw_pair_count"))
        for (distance_a, g_value), (_, count) in zip(rdf, raw, strict=True):
            writer.writerow((f"{distance_a / 10.0:.6f}", f"{g_value:.10f}", f"{count:.0f}"))

    log_text = log_path.read_text() if log_path.is_file() else ""
    volume_match = re.search(r"Average volume is\s+([0-9.Ee+-]+)\s+Ang\^3", log_text)
    frame_match = re.search(r"Read\s+(\d+)\s+frames and processed\s+(\d+)\s+frames", log_text)
    expected_frames = (stop - start) // stride + 1
    frames = int(frame_match.group(1)) if frame_match else expected_frames
    volume = float(volume_match.group(1)) if volume_match else float("nan")
    return {
        "run": run.label,
        "source_arc": str(run.trajectory),
        "temperature_K": run.temperature,
        "frames": frames,
        "molecules": 1000,
        "spacing_A": spacing_a,
        "maximum_A": maximum_a,
        "average_volume_A3": volume,
        "cpptraj_output_reused": str(reused),
        "csv": str(csv_path),
    }


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    cmap = plt.get_cmap("turbo", len(rows))
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    maximum = 0.0
    for row in rows:
        with Path(str(row["csv"])).open(newline="") as handle:
            table = list(csv.DictReader(handle))
        radius = np.asarray([float(item["r_nm"]) for item in table])
        rdf = np.asarray([float(item["g_com_com"]) for item in table])
        curves.append((radius, rdf))
        maximum = max(maximum, float(rdf.max()))

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(11.5, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat = axes.ravel()
    for index, (ax, row, curve) in enumerate(zip(flat, rows, curves, strict=False)):
        radius, rdf = curve
        ax.plot(radius, rdf, color=cmap(index), linewidth=1.5)
        ax.axhline(1.0, color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{float(row['temperature_K']):.0f} K", fontsize=11)
        ax.set_xlim(0.0, float(row["maximum_A"]) / 10.0)
        ax.set_ylim(0.0, 1.08 * maximum)
        ax.grid(alpha=0.18)

    flat[-1].axis("off")
    flat[-1].text(
        0.5,
        0.56,
        "Spherical molecular COM-COM RDF\n"
        "1000 molecules x 1000 frames\n"
        "Cpptraj: autoimage + bymol mass + volume normalization",
        transform=flat[-1].transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
    )
    for row_index in range(4):
        axes[row_index, 0].set_ylabel("g_COM-COM(r)")
    for ax in axes[-1, :2]:
        ax.set_xlabel("COM-COM distance r (nm)")
    fig.suptitle("5CB molecular center-of-mass radial distributions", fontsize=15)
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
        default=AMOEBA_5CB.raw_path(),
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/topology/5cb-x1000-first-frame.pdb"),
    )
    parser.add_argument("--out-data", type=Path, default=Path("data-processed/5cb/amoeba-pt2.7/com-rdf"))
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("reports/amoeba_5cb_com_rdf/com_com_rdf_11panel.png"),
    )
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--spacing-A", type=float, default=0.1)
    parser.add_argument("--maximum-A", type=float, default=35.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-cpptraj", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.data_root.resolve())
    topology = args.topology.resolve()
    if not topology.is_file():
        raise FileNotFoundError(topology)
    cpptraj = shutil.which(args.cpptraj)
    if cpptraj is None:
        raise SystemExit(f"cpptraj executable not found: {args.cpptraj}")
    out_data = args.out_data.resolve()
    out_data.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_run,
                run,
                str(topology),
                str(out_data),
                cpptraj,
                args.start,
                args.stop,
                args.stride,
                args.spacing_A,
                args.maximum_A,
                args.force_cpptraj,
            ): run
            for run in runs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['run']} ({float(row['temperature_K']):.0f} K): "
                f"{row['frames']} frames, reused_cpptraj={row['cpptraj_output_reused']}",
                flush=True,
            )

    rows.sort(key=lambda row: float(row["temperature_K"]))
    summary = out_data / "summary.csv"
    write_summary(rows, summary)
    output_plot = args.out_plot.resolve()
    make_plot(rows, output_plot)
    (out_data / "README.md").write_text(
        "# AMOEBA 5CB molecular COM-COM RDF\n\n"
        "Cpptraj read the 11 external ARC trajectories without modifying them. "
        "For each run it used `autoimage` and calculated the spherical radial "
        "distribution between the mass-weighted centers of all 1,000 molecular "
        "residues with `bymol1 bymol2 mass volume`. The explicit `volume` option "
        "normalizes using the trajectory cell volume, giving `g(r) -> 1` at "
        "large distance. The calculation uses 1,000 frames, 0.1 A bins, and a "
        "35 A maximum. Every run directory retains the cpptraj input, log, "
        "normalized RDF, raw pair counts, and a CSV converted to nm. Existing "
        "cpptraj outputs are reused unless `--force-cpptraj` is given. This is "
        "a spherical positional correlation with no director, cylinder, or "
        "orientational weighting.\n"
    )
    print(f"Wrote {summary}")
    print(f"Wrote {output_plot}")


if __name__ == "__main__":
    main()
