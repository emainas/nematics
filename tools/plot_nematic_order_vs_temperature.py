#!/usr/bin/env python3
"""Plot mean nematic order versus temperature for 4OCB and 5CB."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEADER = struct.Struct("=8siiiqiiii")


@dataclass(frozen=True)
class Result:
    temperature_k: float
    mean_s: float
    frames: int
    path: Path


def find_results(paths: list[Path]) -> list[Path]:
    """Find nematic-order binaries below the supplied paths."""
    found: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            found.update(result.resolve() for result in path.rglob("nematic-order.bin"))
        else:
            raise FileNotFoundError(path)

    if not found:
        raise FileNotFoundError("No nematic-order.bin files found")
    return sorted(found)


def read_result(path: Path, expected_molecule: str) -> Result:
    """Read one binary and calculate <S> from its largest eigenvalue."""
    metadata_path = path.with_suffix(".meta.yaml")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)

    source = metadata["source"]
    molecule = str(source["molecule"]).lower()
    if molecule != expected_molecule.lower():
        raise ValueError(
            f"{path}: metadata says {molecule!r}, expected {expected_molecule!r}"
        )
    temperature_k = float(source["temperature_k"])

    with path.open("rb") as handle:
        raw_header = handle.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise ValueError(f"{path}: incomplete header")

    magic, version, marker, dtype_bytes, frames, nvalues, nvectors, record_size, header_size = (
        HEADER.unpack(raw_header)
    )
    expected = (b"NEMORDR1", 1, 0x01020304, 8, 3, 9, 12, HEADER.size)
    observed = (
        magic,
        version,
        marker,
        dtype_bytes,
        nvalues,
        nvectors,
        record_size,
        header_size,
    )
    if observed != expected or frames <= 0:
        raise ValueError(f"{path}: unsupported nematic-order binary header")

    expected_bytes = HEADER.size + frames * record_size * dtype_bytes
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: file size does not match its header")

    records = np.memmap(
        path,
        mode="r",
        dtype=np.dtype("=f8"),
        offset=HEADER.size,
        shape=(frames, record_size),
    )
    s_per_frame = np.max(records[:, :3], axis=1)
    if not np.all(np.isfinite(s_per_frame)):
        raise ValueError(f"{path}: non-finite eigenvalue found")

    return Result(temperature_k, float(np.mean(s_per_frame)), frames, path)


def load_series(paths: list[Path], molecule: str) -> list[Result]:
    results = [read_result(path, molecule) for path in find_results(paths)]
    return sorted(results, key=lambda result: (result.temperature_k, str(result.path)))


def plot_series(ax: plt.Axes, results: list[Result], label: str, color: str) -> None:
    temperatures = [result.temperature_k for result in results]
    mean_s = [result.mean_s for result in results]
    ax.plot(temperatures, mean_s, color=color, linewidth=1.8)
    ax.scatter(temperatures, mean_s, color=color, s=38, label=label, zorder=3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate <S> from nematic-order binaries and plot it versus "
            "temperature for 4OCB and 5CB."
        )
    )
    parser.add_argument(
        "--4ocb",
        dest="paths_4ocb",
        type=Path,
        nargs="+",
        required=True,
        help="4OCB nematic-order binary, run directory, or dataset directory",
    )
    parser.add_argument(
        "--5cb",
        dest="paths_5cb",
        type=Path,
        nargs="+",
        required=True,
        help="5CB nematic-order binary, run directory, or dataset directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/nematic_order/s_vs_temperature.png"),
        help="Output figure path",
    )
    args = parser.parse_args()

    series_4ocb = load_series(args.paths_4ocb, "4ocb")
    series_5cb = load_series(args.paths_5cb, "5cb")

    figure, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    plot_series(ax, series_4ocb, "4OCB", "#0072B2")
    plot_series(ax, series_5cb, "5CB", "#D55E00")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Mean nematic order, $\langle S \rangle$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)

    print("system,temperature_k,mean_S,frames,path")
    for label, results in (("4OCB", series_4ocb), ("5CB", series_5cb)):
        for result in results:
            print(
                f"{label},{result.temperature_k:g},{result.mean_s:.8f},"
                f"{result.frames},{result.path}"
            )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
