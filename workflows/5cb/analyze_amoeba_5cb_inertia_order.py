#!/usr/bin/env python3
"""Calculate AMOEBA 5CB nematic order from molecular inertia axes.

For every molecule and frame, cpptraj constructs the mass-weighted inertia
tensor about the molecular center of mass, diagonalizes it, and returns its
``principal z`` vector. Cpptraj sorts principal moments in descending order,
so ``principal z`` is the eigenvector of the smallest principal moment and is
used here as the molecular long axis.

The raw cpptraj vector tables are permanent analysis products. Existing tables
are reused by default; use ``--force-cpptraj`` only to replace them.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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


def discover_runs(data_root: Path) -> list[Run]:
    return [
        Run(run.id, run.temperature_k, trajectory.resolve())
        for run, trajectory in AMOEBA_5CB.run_trajectories(data_root)
    ]


def write_cpptraj_input(
    path: Path,
    topology: Path,
    trajectory: Path,
    raw_axes: Path,
    nmolecules: int,
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
        f"vector inertia_z_{residue:04d} principal z :{residue} "
        f"out {quote_cpptraj(raw_axes)}"
        for residue in range(1, nmolecules + 1)
    )
    lines.extend(
        [
            f"precision {quote_cpptraj(raw_axes)} 14 8",
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


def load_principal_axes(path: Path, nmolecules: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", ndmin=2)
    frames = table[:, 0].astype(int)
    values = table[:, 1:]
    expected = 6 * nmolecules
    if values.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} cpptraj vector fields in {path}, "
            f"found {values.shape[1]}"
        )
    # Each cpptraj vector record is Vx,Vy,Vz,Ox,Oy,Oz. For a principal
    # vector, V is the unit eigenvector and O is the molecular center of mass.
    records = values.reshape(values.shape[0], nmolecules, 6)
    axes = records[:, :, :3]
    centers = records[:, :, 3:]
    if not np.isfinite(records).all():
        raise ValueError(f"Non-finite cpptraj principal-axis result in {path}")
    return frames, axes, centers


def frame_order(
    axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(axes, axis=2)
    if np.any(norms == 0.0):
        raise ValueError("Zero-length principal axis")
    unit = axes / norms[:, :, None]
    nmolecules = unit.shape[1]
    q = 1.5 * np.einsum("fmi,fmj->fij", unit, unit) / nmolecules
    q -= 0.5 * np.eye(3)[None, :, :]
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    s_values = eigenvalues[:, -1]
    directors = eigenvectors[:, :, -1]
    return s_values, directors, q, eigenvalues, norms


def analyze_one_run(
    run: Run,
    topology: str,
    out_root: str,
    cpptraj: str,
    nmolecules: int,
    start: int,
    stop: int,
    stride: int,
    force_cpptraj: bool,
) -> dict[str, float | int | str]:
    out_dir = Path(out_root) / run.label
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_axes = out_dir / "inertia_principal_z_vectors.dat"
    input_path = out_dir / "inertia_principal_z.cpptraj.in"
    log_path = out_dir / "inertia_principal_z.cpptraj.log"
    write_cpptraj_input(
        input_path,
        Path(topology),
        run.trajectory,
        raw_axes,
        nmolecules,
        start,
        stop,
        stride,
    )
    reused = raw_axes.is_file() and not force_cpptraj
    if not reused:
        run_cpptraj(cpptraj, input_path, log_path)

    frames, axes, centers = load_principal_axes(raw_axes, nmolecules)
    s_values, directors, q_tensors, eigenvalues, norms = frame_order(axes)
    ddof = 1 if s_values.size > 1 else 0
    std_s = float(s_values.std(ddof=ddof))

    np.save(out_dir / "principal_z_vectors.npy", axes)
    np.save(out_dir / "centers_of_mass.npy", centers)
    np.save(out_dir / "S_by_frame.npy", s_values)
    np.save(out_dir / "director_by_frame.npy", directors)
    np.save(out_dir / "Q_by_frame.npy", q_tensors)
    np.save(out_dir / "Q_eigenvalues_by_frame.npy", eigenvalues)
    np.savetxt(
        out_dir / "S_by_frame.csv",
        np.column_stack((frames, s_values, directors, np.trace(q_tensors, axis1=1, axis2=2))),
        delimiter=",",
        header="frame,S,n_x,n_y,n_z,Tr_Q",
        comments="",
        fmt=("%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.12e"),
    )
    return {
        "run": run.label,
        "source_arc": str(run.trajectory),
        "temperature_K": run.temperature,
        "frames": int(s_values.size),
        "molecules": nmolecules,
        "mean_S": float(s_values.mean()),
        "std_S": std_s,
        "sem_S": float(std_s / np.sqrt(s_values.size)),
        "mean_axis_norm": float(norms.mean()),
        "min_axis_norm": float(norms.min()),
        "max_axis_norm": float(norms.max()),
        "max_abs_Q_trace": float(np.max(np.abs(np.trace(q_tensors, axis1=1, axis2=2)))),
        "cpptraj_output_reused": str(reused),
    }


def write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_comparison(
    inertia_rows: list[dict[str, float | int | str]],
    end_to_end_summary: Path,
    output_csv: Path,
    output_plot: Path,
) -> None:
    end_rows = {
        float(row["temperature_K"]): row for row in read_summary(end_to_end_summary)
    }
    joined: list[dict[str, float | int | str]] = []
    for inertia in inertia_rows:
        temperature = float(inertia["temperature_K"])
        if temperature not in end_rows:
            raise ValueError(f"No end-to-end result at {temperature:g} K")
        end = end_rows[temperature]
        joined.append(
            {
                "temperature_K": temperature,
                "frames": int(inertia["frames"]),
                "molecules": int(inertia["molecules"]),
                "mean_S_inertia": float(inertia["mean_S"]),
                "sem_S_inertia": float(inertia["sem_S"]),
                "mean_S_end_to_end": float(end["mean_S"]),
                "sem_S_end_to_end": float(end["sem_S"]),
                "delta_S_inertia_minus_end_to_end": (
                    float(inertia["mean_S"]) - float(end["mean_S"])
                ),
            }
        )
    write_summary(joined, output_csv)

    import matplotlib.pyplot as plt

    temperature = np.asarray([float(row["temperature_K"]) for row in joined])
    inertia_mean = np.asarray([float(row["mean_S_inertia"]) for row in joined])
    inertia_sem = np.asarray([float(row["sem_S_inertia"]) for row in joined])
    end_mean = np.asarray([float(row["mean_S_end_to_end"]) for row in joined])
    end_sem = np.asarray([float(row["sem_S_end_to_end"]) for row in joined])

    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    ax.errorbar(
        temperature,
        end_mean,
        yerr=end_sem,
        color="#d95f02",
        marker="o",
        linewidth=2,
        capsize=3,
        label="End-to-end axis (C18 to N01)",
    )
    ax.errorbar(
        temperature,
        inertia_mean,
        yerr=inertia_sem,
        color="#1f9e89",
        marker="s",
        linewidth=2,
        capsize=3,
        label="Inertia axis (smallest principal moment)",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean nematic order, <S>")
    ax.set_xticks(temperature)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)


def write_readme(
    runs: list[Run],
    rows: list[dict[str, float | int | str]],
    out_data: Path,
    comparison_csv: Path,
    comparison_plot: Path,
    topology: Path,
) -> None:
    sources = "\n".join(
        f"- `{run.label}` ({run.temperature:.0f} K): `{run.trajectory}`" for run in runs
    )
    results = "\n".join(
        f"- {float(row['temperature_K']):.0f} K: `<S> = {float(row['mean_S']):.8f}`"
        for row in rows
    )
    (out_data / "README.md").write_text(
        "# AMOEBA 5CB inertia-axis nematic order\n\n"
        "The external Tinker ARC files were read only. The cpptraj topology was "
        f"`{topology}`. Cpptraj first used `autoimage`, then for every residue "
        "and frame calculated the mass-weighted molecular inertia tensor about "
        "the center of mass and diagonalized it. Cpptraj sorts the principal "
        "moments from largest to smallest; therefore `vector ... principal z` "
        "is the eigenvector belonging to the smallest principal moment and was "
        "used as the apolar molecular long axis. PDB element-derived masses were "
        "used by cpptraj.\n\n"
        "The raw `inertia_principal_z_vectors.dat` files are cpptraj outputs. "
        "Each molecular record contains the three eigenvector components and "
        "the three coordinates of its center of mass. These files are retained "
        "and reused automatically, so cpptraj does not need to be rerun. The "
        "eigenvector sign is arbitrary but cancels in the second-rank tensor.\n\n"
        "For every frame:\n\n"
        "```text\n"
        "Q = (1/N) sum_i [ (3/2) u_i u_i - (1/2) I ]\n"
        "S_frame = largest eigenvalue of Q\n"
        "<S> = mean over frames of S_frame\n"
        "```\n\n"
        "The error bars are naive standard errors over frames and do not correct "
        "for temporal autocorrelation.\n\n"
        "## Read-only sources\n\n"
        f"{sources}\n\n"
        "## Inertia-axis results\n\n"
        f"{results}\n\n"
        "## Main outputs\n\n"
        f"- Inertia summary: `{out_data / 'summary.csv'}`\n"
        f"- Joined comparison data: `{comparison_csv}`\n"
        f"- Comparison plot: `{comparison_plot}`\n"
    )


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
    parser.add_argument(
        "--end-to-end-summary",
        type=Path,
        default=Path("data-processed/5cb/amoeba-pt2.7/end-to-end-order/summary.csv"),
    )
    parser.add_argument(
        "--out-data", type=Path, default=Path("data-processed/5cb/amoeba-pt2.7/inertia-order")
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path(
            "reports/amoeba_5cb_inertia_order/s_vs_temperature_inertia_vs_end_to_end.png"
        ),
    )
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--nmolecules", type=int, default=1000)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force-cpptraj",
        action="store_true",
        help="Replace saved raw cpptraj tables instead of reusing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.data_root.resolve())
    topology = args.topology.resolve()
    end_summary = args.end_to_end_summary.resolve()
    if not topology.is_file():
        raise FileNotFoundError(topology)
    if not end_summary.is_file():
        raise FileNotFoundError(end_summary)
    cpptraj = shutil.which(args.cpptraj)
    if cpptraj is None:
        raise SystemExit(f"cpptraj executable not found: {args.cpptraj}")

    out_data = args.out_data.resolve()
    out_data.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_one_run,
                run,
                str(topology),
                str(out_data),
                cpptraj,
                args.nmolecules,
                args.start,
                args.stop,
                args.stride,
                args.force_cpptraj,
            ): run
            for run in runs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['run']} ({float(row['temperature_K']):.0f} K): "
                f"{row['frames']} frames, <S>={float(row['mean_S']):.6f}, "
                f"reused_cpptraj={row['cpptraj_output_reused']}",
                flush=True,
            )

    rows.sort(key=lambda row: float(row["temperature_K"]))
    inertia_summary = out_data / "summary.csv"
    comparison_csv = out_data / "comparison_with_end_to_end.csv"
    output_plot = args.out_plot.resolve()
    write_summary(rows, inertia_summary)
    write_comparison(rows, end_summary, comparison_csv, output_plot)
    write_readme(
        runs, rows, out_data, comparison_csv, output_plot, topology
    )
    print(f"Wrote {inertia_summary}")
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {output_plot}")


if __name__ == "__main__":
    main()
