#!/usr/bin/env python3
"""Digitize the 5CB curves on pages 5--8 of try_to_reproduce.pdf.

Pages 5--7 contain vector paths, which are recovered directly from SVG output.
Page 8 contains one embedded raster image; its purple curve is sampled at each
image column using the median center of the exact curve-color pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nematics.datasets import get_dataset  # noqa: E402

GAFF_RUNS = tuple(reversed(get_dataset("gaff-5cb").runs))
TEMPERATURES = tuple(int(run.temperature_k) for run in GAFF_RUNS)
RUN_FOR_TEMPERATURE = {int(run.temperature_k): run.id for run in GAFF_RUNS}
SVG_NS = "{http://www.w3.org/2000/svg}"
CURVE_COLORS = {
    "magenta": "rgb(58.039856%, 0%, 82.745361%)",
    "green": "rgb(0%, 61.961365%, 45.097351%)",
    "cyan": "rgb(33.724976%, 70.588684%, 91.372681%)",
}
COLOR_TO_RADIUS = {"magenta": 1, "green": 2, "cyan": 3}
X_EDGES = np.array([51.666944, 164.998667, 278.330390, 392.002005, 505.329255])

# Two tick anchors per row give a more accurate calibration than reading the
# axis rectangle. Coordinates are in the untransformed plotting coordinate
# system stored in the PDF vector paths.
VECTOR_Y_CALIBRATION = {
    5: {
        0: ((262.998601, 0.8), (360.001778, 1.1)),
        1: ((161.669958, -0.1), (199.668758, 0.0)),
        2: ((39.331266, 0.0), (137.998428, 0.6)),
    },
    6: {
        0: ((262.999952, 0.8), (359.998655, 1.1)),
        1: ((161.671308, -0.1), (199.670109, 0.0)),
        2: ((39.332616, 0.0), (137.999778, 0.6)),
    },
    7: {
        0: ((253.329156, 0.0), (359.998655, 1.6)),
        1: ((159.671842, -0.1), (186.331389, 0.0)),
        2: ((39.332616, 0.0), (146.328650, 1.0)),
    },
}


def linear_map(value: np.ndarray | float, anchor1, anchor2):
    x1, y1 = anchor1
    x2, y2 = anchor2
    return y1 + (np.asarray(value) - x1) * (y2 - y1) / (x2 - x1)


def parse_path_points(path_data: str) -> np.ndarray:
    numbers = [
        float(value)
        for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path_data)
    ]
    if len(numbers) % 2:
        raise ValueError("Unexpected odd coordinate count in SVG path")
    return np.asarray(numbers, dtype=float).reshape(-1, 2)


def curve_row(points: np.ndarray, page: int) -> int:
    mean_y = float(np.mean(points[:, 1]))
    if page in (5, 6):
        return 0 if mean_y > 253.0 else (1 if mean_y > 146.0 else 2)
    return 0 if mean_y > 253.0 else (1 if mean_y > 146.0 else 2)


def curve_column(points: np.ndarray) -> int:
    mean_x = float(np.mean(points[:, 0]))
    column = int(np.searchsorted(X_EDGES[1:], mean_x, side="right"))
    return min(max(column, 0), 3)


def vector_x(points_x: np.ndarray, column: int, page: int) -> np.ndarray:
    left, right = X_EDGES[column], X_EDGES[column + 1]
    if page in (5, 6):
        return -5.0 + 10.0 * (points_x - left) / (right - left)
    return 5.0 * (points_x - left) / (right - left)


def extract_vector_page(svg_path: Path, page: int) -> list[dict[str, object]]:
    root = ET.parse(svg_path).getroot()
    records: list[dict[str, object]] = []
    segment_counter: defaultdict[tuple[int, int, str], int] = defaultdict(int)
    for element in root.iter(f"{SVG_NS}path"):
        color_name = next(
            (name for name, value in CURVE_COLORS.items() if element.get("stroke") == value),
            None,
        )
        if color_name is None:
            continue
        points = parse_path_points(element.get("d", ""))
        if len(points) <= 5:  # legend swatches and zero reference lines
            continue
        row = curve_row(points, page)
        column = curve_column(points)
        key = (row, column, color_name)
        segment_id = segment_counter[key]
        segment_counter[key] += 1
        x = vector_x(points[:, 0], column, page)
        y = linear_map(points[:, 1], *VECTOR_Y_CALIBRATION[page][row])
        for point_index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
            record: dict[str, object] = {
                "temperature_K": TEMPERATURES[column],
                "order": row,
                "coordinate_nm": float(x_value),
                "value": float(y_value),
                "segment_id": segment_id,
                "point_index": point_index,
                "source_page": page,
                "extraction": "PDF vector path",
            }
            if page in (5, 6):
                record["radius_nm"] = COLOR_TO_RADIUS[color_name]
            else:
                record["axis_reference"] = (
                    "global_director" if color_name == "magenta" else "molecular_z"
                )
            records.append(record)
    return records


def write_records(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def extract_page8(image_path: Path) -> list[dict[str, object]]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    purple = np.all(image == np.array([148, 0, 211], dtype=np.uint8), axis=2)
    x_edges = np.array([111.5, 365.5, 619.5, 874.0, 1128.0])
    y_calibration = {
        0: ((67.0, 1.2), (306.0, 0.65)),
        1: ((306.0, 0.15), (546.0, -0.25)),
        2: ((546.0, 0.75), (786.0, 0.0)),
    }
    records: list[dict[str, object]] = []
    for column, temperature in enumerate(TEMPERATURES):
        left, right = x_edges[column], x_edges[column + 1]
        for order, anchors in y_calibration.items():
            y_low = int(np.floor(min(anchors[0][0], anchors[1][0])))
            y_high = int(np.ceil(max(anchors[0][0], anchors[1][0])))
            first_x = int(np.ceil(left + 2))
            last_x = int(np.floor(right - 20))
            rows_by_x = {
                pixel_x: np.flatnonzero(purple[y_low : y_high + 1, pixel_x]) + y_low
                for pixel_x in range(first_x, last_x + 1)
            }
            rows_by_x = {x: rows for x, rows in rows_by_x.items() if rows.size}
            previous_x: int | None = None
            segment_id = 0
            point_index = 0
            # The source plotting polygon closes against the right-hand frame
            # during its last few pixels; stop before that non-data edge.
            for pixel_x, pixel_rows in rows_by_x.items():
                pixel_spread = int(pixel_rows.max() - pixel_rows.min())
                if previous_x is not None and pixel_x - previous_x > 1:
                    segment_id += 1
                pixel_y_median = float(np.median(pixel_rows))
                selected_rows = [pixel_y_median]
                for pixel_y in selected_rows:
                    records.append(
                        {
                            "temperature_K": temperature,
                            "order": order,
                            "coordinate_nm": float(5.0 * (pixel_x - left) / (right - left)),
                            "value": float(linear_map(pixel_y, *anchors)),
                            "segment_id": segment_id,
                            "point_index": point_index,
                            "pixel_x": pixel_x,
                            "pixel_y_selected": pixel_y,
                            "pixel_y_median": pixel_y_median,
                            "pixel_y_spread": pixel_spread,
                            "source_page": 8,
                            "extraction": "embedded raster; curve-color centerline/endpoints",
                        }
                    )
                    point_index += 1
                previous_x = pixel_x
    return records


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select(records, **conditions):
    selected = []
    for record in records:
        if all(str(record[key]) == str(value) for key, value in conditions.items()):
            selected.append(record)
    return selected


def plot_segments(axis, records, color: str, label: str | None = None, **kwargs) -> None:
    segment_ids = sorted({int(record["segment_id"]) for record in records})
    for index, segment_id in enumerate(segment_ids):
        segment = [record for record in records if int(record["segment_id"]) == segment_id]
        segment.sort(key=lambda record: int(record.get("point_index", record.get("pixel_x", 0))))
        axis.plot(
            [float(record["coordinate_nm"]) for record in segment],
            [float(record["value"]) for record in segment],
            color=color,
            label=label if index == 0 else None,
            **kwargs,
        )


def scatter_records(axis, records, color: str, label: str | None = None) -> None:
    """Plot extracted reference values as circles without connecting lines."""
    ordered = sorted(
        records,
        key=lambda record: (
            int(record["segment_id"]),
            int(record.get("point_index", record.get("pixel_x", 0))),
        ),
    )
    axis.scatter(
        [float(record["coordinate_nm"]) for record in ordered],
        [float(record["value"]) for record in ordered],
        color=color,
        s=7,
        marker="o",
        linewidths=0,
        alpha=0.75,
        label=label,
        zorder=3,
    )


def centers(minimum: float, maximum: float, size: int) -> np.ndarray:
    width = (maximum - minimum) / size
    return minimum + width * (np.arange(size) + 0.5)


def load_ours(data_root: Path, temperature: int) -> dict[str, np.ndarray]:
    label = RUN_FOR_TEMPERATURE[temperature]
    with np.load(data_root / label / "correlations.npz") as saved:
        nsnaps = int(saved["nsnaps"])
        return {
            name: np.asarray(saved[name], dtype=float) / nsnaps
            for name in (
                "global_par_1", "global_par_2", "global_par_3",
                "local_par_1", "local_par_2", "local_par_3",
                "global_per", "local_per", "cn_rad",
            )
        }


def alekos_report_channels(path: Path) -> np.ndarray:
    """Load Alekos output and return its plotted g0, g1, and P2 channels."""
    raw = np.loadtxt(path)
    return np.column_stack(
        (
            raw[:, 2],
            raw[:, 3],
            (3.0 * raw[:, 4] - raw[:, 2]) / 2.0,
        )
    )


def load_alekos_exact(exact_root: Path, temperature: int) -> dict[str, np.ndarray]:
    run = RUN_FOR_TEMPERATURE[temperature]
    run_dir = exact_root / run
    return {
        "global_par_1": alekos_report_channels(run_dir / f"{temperature}_5CB_par_1.0.dat"),
        "global_par_2": alekos_report_channels(run_dir / f"{temperature}_5CB_par_2.0.dat"),
        "global_par_3": alekos_report_channels(run_dir / f"{temperature}_5CB_par_3.0.dat"),
        "global_per": alekos_report_channels(run_dir / f"{temperature}_5CB_per.dat"),
        "cn_rad": alekos_report_channels(run_dir / f"{temperature}_CN_rad.dat"),
    }


def style_axis(axis, xlim, ylim):
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.axhline(0.0, color="0.7", linewidth=0.6)
    axis.grid(axis="x", color="0.75", linestyle=":", linewidth=0.65)
    axis.tick_params(direction="in", top=True, right=True)


def compare_parallel(
    data_root: Path,
    exact_root: Path,
    alekos_csv: Path,
    out_path: Path,
    local: bool,
) -> None:
    alekos = read_records(alekos_csv)
    our_colors = {1: "#7b2cbf", 2: "#009e73", 3: "#289dc1"}
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 7.6), sharex=True, sharey="row")
    for column, temperature in enumerate(TEMPERATURES):
        ours = load_ours(data_root, temperature)
        exact = load_alekos_exact(exact_root, temperature)
        for radius in (1, 2, 3):
            array = (
                ours[f"local_par_{radius}"]
                if local
                else exact[f"global_par_{radius}"]
            )
            if local:
                x = centers(-5.0, 5.0, array.shape[0])
                values = array[:, 1:4]
            else:
                positive_x = centers(0.0, 5.0, array.shape[0])
                x = np.concatenate((-positive_x[::-1], positive_x))
                values = np.concatenate((array[::-1, :], array), axis=0)
            for order in range(3):
                axes[order, column].plot(
                    x, values[:, order], color=our_colors[radius], linewidth=1.0,
                    linestyle="--", alpha=0.9,
                    label=("reconstructed" if local else "exact rerun") + f" R={radius} nm",
                )
                subset = select(
                    alekos,
                    temperature_K=temperature,
                    order=order,
                    radius_nm=radius,
                )
                scatter_records(
                    axes[order, column], subset, our_colors[radius],
                    f"report R={radius} nm",
                )
        axes[0, column].set_title(f"{temperature} K")
    ylims = ((0.77, 1.10), (-0.14, 0.14), (0.0, 0.65))
    for order in range(3):
        for column in range(4):
            style_axis(axes[order, column], (-5.0, 5.0), ylims[order])
        axes[order, 0].set_ylabel(fr"$g_{order}^{{\parallel}}$")
    for axis in axes[-1]:
        axis.set_xlabel("z (nm)")
    handles, labels = axes[2, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[2, 0].legend(unique.values(), unique.keys(), frameon=False, fontsize=7, ncol=2)
    qualifier = "molecular z-axis" if local else "nematic director"
    comparison = "reconstruction vs report" if local else "exact Alekos rerun vs report"
    fig.suptitle(f"5CB correlations along the {qualifier}: {comparison}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0, w_pad=0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compare_perpendicular(
    data_root: Path, exact_root: Path, alekos_csv: Path, out_path: Path
) -> None:
    alekos = read_records(alekos_csv)
    styles = {
        "global_director": ("global_per", "#7b2cbf", "global n"),
        "molecular_z": ("local_per", "#009e73", "molecular z"),
    }
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 7.6), sharex=True, sharey="row")
    x = centers(0.0, 5.0, 200)
    for column, temperature in enumerate(TEMPERATURES):
        ours = load_ours(data_root, temperature)
        exact = load_alekos_exact(exact_root, temperature)
        for reference, (key, color, short_label) in styles.items():
            for order in range(3):
                values = (
                    exact["global_per"]
                    if reference == "global_director"
                    else ours[key][:, 1:4]
                )
                axes[order, column].plot(
                    x, values[:, order], color=color, linewidth=1.0,
                    linestyle="--",
                    label=("exact rerun: " if reference == "global_director" else "reconstructed: ")
                    + short_label,
                )
                subset = select(
                    alekos,
                    temperature_K=temperature,
                    order=order,
                    axis_reference=reference,
                )
                scatter_records(
                    axes[order, column], subset, color,
                    f"report: {short_label}",
                )
        axes[0, column].set_title(f"{temperature} K")
    ylims = ((0.0, 1.6), (-0.15, 0.25), (0.0, 1.0))
    for order in range(3):
        for column in range(4):
            style_axis(axes[order, column], (0.0, 5.0), ylims[order])
        axes[order, 0].set_ylabel(fr"$g_{order}^{{\perp}}$")
    for axis in axes[-1]:
        axis.set_xlabel(r"$r_\perp$ (nm)")
    handles, labels = axes[2, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[2, 0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    fig.suptitle("5CB perpendicular correlations: rerun/reconstruction vs report", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0, w_pad=0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compare_cn(exact_root: Path, alekos_csv: Path, out_path: Path) -> None:
    alekos = read_records(alekos_csv)
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 7.6), sharex=True, sharey="row")
    x = centers(0.0, 5.0, 200)
    for column, temperature in enumerate(TEMPERATURES):
        exact = load_alekos_exact(exact_root, temperature)["cn_rad"]
        for order in range(3):
            axes[order, column].plot(
                x, exact[:, order], color="#7b2cbf", linewidth=1.1,
                linestyle="--", label="exact rerun",
            )
            subset = select(alekos, temperature_K=temperature, order=order)
            scatter_records(
                axes[order, column], subset, "#7b2cbf", "report digitized",
            )
        axes[0, column].set_title(f"{temperature} K")
    ylims = ((0.65, 1.2), (-0.25, 0.15), (0.0, 0.75))
    for order in range(3):
        for column in range(4):
            style_axis(axes[order, column], (0.0, 5.0), ylims[order])
        axes[order, 0].set_ylabel(fr"$g_{order}^{{\mathrm{{rad}}}}$")
    for axis in axes[-1]:
        axis.set_xlabel("r (nm)")
    axes[2, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("CN dipole radial correlations: exact Alekos rerun vs report", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0, w_pad=0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument("--out-data", required=True, type=Path)
    parser.add_argument("--out-plots", required=True, type=Path)
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    data_root = args.data_root.resolve()
    exact_root = args.exact_root.resolve()
    out_data = args.out_data.resolve()
    out_plots = args.out_plots.resolve()
    out_data.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="alekos_digitize_") as temporary:
        temporary_path = Path(temporary)
        vector_records = {}
        for page in (5, 6, 7):
            svg_path = temporary_path / f"page{page}.svg"
            subprocess.run(
                ["pdftocairo", "-f", str(page), "-l", str(page), "-svg", str(pdf), str(svg_path)],
                check=True,
            )
            vector_records[page] = extract_vector_page(svg_path, page)

        image_prefix = temporary_path / "page8"
        subprocess.run(
            ["pdfimages", "-f", "8", "-l", "8", "-png", str(pdf), str(image_prefix)],
            check=True,
        )
        page8_images = sorted(temporary_path.glob("page8-*.png"))
        if len(page8_images) != 1:
            raise RuntimeError(f"Expected one page-8 image, found {len(page8_images)}")
        page8_image = page8_images[0]
        page8_records = extract_page8(page8_image)
        shutil.copy2(page8_image, out_data / "source_page8_embedded.png")

    page5_csv = out_data / "alekos_page5_global_parallel.csv"
    page6_csv = out_data / "alekos_page6_local_parallel.csv"
    page7_csv = out_data / "alekos_page7_perpendicular.csv"
    page8_csv = out_data / "alekos_page8_cn_radial.csv"
    parallel_fields = [
        "temperature_K", "order", "radius_nm", "coordinate_nm", "value",
        "segment_id", "point_index", "source_page", "extraction",
    ]
    write_records(page5_csv, vector_records[5], parallel_fields)
    write_records(page6_csv, vector_records[6], parallel_fields)
    write_records(
        page7_csv,
        vector_records[7],
        [
            "temperature_K", "order", "axis_reference", "coordinate_nm", "value",
            "segment_id", "point_index", "source_page", "extraction",
        ],
    )
    write_records(
        page8_csv,
        page8_records,
        [
            "temperature_K", "order", "coordinate_nm", "value", "segment_id",
            "point_index", "pixel_x", "pixel_y_selected", "pixel_y_median",
            "pixel_y_spread", "source_page", "extraction",
        ],
    )

    metadata = {
        "source_pdf": str(pdf),
        "pages": {
            "5": "vector paths; global parallel",
            "6": "vector paths; local parallel",
            "7": "vector paths; perpendicular global/local; values outside plot limits are clipped in source",
            "8": (
                "embedded 1149x871 raster; median centerline of RGB(148,0,211) pixels; "
                "pixel y-spread retained as an uncertainty diagnostic; final 20 border pixels omitted"
            ),
        },
        "temperature_order_K": list(TEMPERATURES),
        "page8_estimated_digitization_uncertainty": {
            "coordinate_nm": "+/- 0.01",
            "value": "approximately +/- 0.01, depending on curve thickness",
        },
    }
    (out_data / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    compare_parallel(
        data_root, exact_root, page5_csv,
        out_plots / "5cb_along_nematic_director_comparison.png", local=False,
    )
    compare_parallel(
        data_root, exact_root, page6_csv,
        out_plots / "5cb_along_molecular_z_comparison.png", local=True,
    )
    compare_perpendicular(
        data_root, exact_root, page7_csv,
        out_plots / "5cb_perpendicular_comparison.png",
    )
    compare_cn(
        exact_root, page8_csv,
        out_plots / "5cb_cn_dipole_radial_comparison.png",
    )
    for path in (page5_csv, page6_csv, page7_csv, page8_csv):
        print(f"{path}: {sum(1 for _ in path.open()) - 1} points")
    for path in sorted(out_plots.glob("*_comparison.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
