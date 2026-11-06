#!/usr/bin/env python3
"""Build the concise AMOEBA-only 4OCB results report and DOCX."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_BREAK = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing required production products:\n" + "\n".join(missing))


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def rdf_metrics(summary: list[dict[str, str]]) -> dict[int, tuple[float, float]]:
    result = {}
    for row in summary:
        table = np.loadtxt(row["csv"], delimiter=",", skiprows=1)
        selected = (table[:, 0] >= 0.35) & (table[:, 0] <= 0.80)
        indices = np.where(selected)[0]
        peak = indices[np.argmax(table[selected, 1])]
        result[int(float(row["temperature_K"]))] = (
            float(table[peak, 0]),
            float(table[peak, 1]),
        )
    return result


def torsion_metrics(
    summary: list[dict[str, str]],
) -> dict[int, tuple[float, float, float, float, float]]:
    result = {}
    for row in summary:
        table = np.loadtxt(row["histogram_csv"], delimiter=",", skiprows=1)
        left_indices = np.where((table[:, 0] >= 40.0) & (table[:, 0] <= 120.0))[0]
        right_indices = np.where((table[:, 0] >= 240.0) & (table[:, 0] <= 320.0))[0]
        left = left_indices[np.argmax(table[left_indices, 1])]
        right = right_indices[np.argmax(table[right_indices, 1])]
        result[int(float(row["temperature_K"]))] = (
            float(row["mode_deg_0_360"]),
            float(table[left, 0]),
            float(table[left, 2]),
            float(table[right, 0]),
            float(table[right, 2]),
        )
    return result


def ring_torsion_metrics(
    summary: list[dict[str, str]],
) -> dict[int, tuple[float, float, float, float, float, float]]:
    result = {}
    for row in summary:
        table = np.loadtxt(row["histogram_csv"], delimiter=",", skiprows=1)
        near_zero = (table[:, 0] <= 30.0) | (table[:, 0] >= 330.0)
        near_180 = (table[:, 0] >= 150.0) & (table[:, 0] <= 210.0)
        near_90 = (table[:, 0] >= 60.0) & (table[:, 0] <= 120.0)
        near_270 = (table[:, 0] >= 240.0) & (table[:, 0] <= 300.0)
        zero_indices = np.where(near_zero)[0]
        trans_indices = np.where(near_180)[0]
        zero = zero_indices[np.argmax(table[zero_indices, 1])]
        trans = trans_indices[np.argmax(table[trans_indices, 1])]
        result[int(float(row["temperature_K"]))] = (
            float(table[zero, 0]),
            float(table[zero, 2]),
            float(table[trans, 0]),
            float(table[trans, 2]),
            float(np.nanmax(table[near_90, 2])),
            float(np.nanmax(table[near_270, 2])),
        )
    return result


def image(path: str, width: str = "6.35in") -> str:
    return f"![]({path}){{width={width}}}"


def build_markdown(data_root: Path) -> str:
    order = records(data_root / "order" / "summary.csv")
    rdf = records(data_root / "com-rdf" / "summary.csv")
    torsion = records(data_root / "junction-dihedral" / "summary.csv")
    ring_torsion = records(data_root / "ring-junction-dihedral" / "summary.csv")
    rdf_values = rdf_metrics(rdf)
    torsion_values = torsion_metrics(torsion)
    ring_torsion_values = ring_torsion_metrics(ring_torsion)

    lines = [
        '::: {custom-style="Title"}',
        "Molecular order, pair correlations, and butoxy-junction conformations of 4OCB",
        ":::",
        "",
        '::: {custom-style="Subtitle"}',
        "AMOEBA molecular-dynamics analysis",
        ":::",
        "",
        '::: {custom-style="Author"}',
        "Nematics analysis workflow  |  6 November 2026",
        ":::",
        "",
        "# Scope and principal findings",
        "",
        "This report analyzes the 11 AMOEBA/Tinker 4OCB trajectories at "
        "288, 296, 304, 312, 320, 324, 326, 328, 330, 332, and 334 K. "
        "Each trajectory contains 1,000 frames of 1,000 molecules with 36 "
        "atoms per molecule. Each one-body distribution therefore contains "
        "1,000,000 molecule-frames per temperature. GAFF, legacy Alekos "
        "comparisons, and dipole-specific analysis are outside the present scope.",
        "",
        "The main results are:",
        "",
        "- All three molecular-axis definitions give the same order trend. "
        "The end-to-end result decreases from 0.620 at 320 K to 0.475, 0.335, "
        "0.137, and 0.061 at 324, 326, 328, and 330 K.",
        "- The 332 and 334 K runs rise again to 0.121 and 0.169. Their "
        "frame-block means show that this is persistent within those runs, not "
        "a single-frame artifact. Because these are independent trajectories, "
        "the non-monotonic high-temperature points should not be interpreted "
        "as an equilibrium transition curve without a separate convergence study.",
        "- The first COM--COM shell weakens from a peak of 1.51 at 0.535 nm "
        "at 288 K to approximately 1.09--1.12 at high temperature, while the "
        "long-range RDF approaches one.",
        "- First-rank correlations contain signed short-range packing features "
        "and tend toward zero at long range. Second-rank correlations retain "
        "a long-range plateau in ordered runs and diminish with the loss of order.",
        "- The C-C-O-C(ar) ether torsion has a dominant trans-like basin near "
        "180 degrees and weaker gauche-like basins near 73--80 and 282--288 "
        "degrees, about 1.2--1.6 kcal/mol above the global minimum.",
        "- The separate aryl-O ring-junction torsion has nearly degenerate "
        "coplanar basins near 0 and 180 degrees. Rotation through the approximately "
        "normal arrangements at 90 and 270 degrees costs about 2.7--3.0 kcal/mol.",
        "",
        "```{=openxml}",
        PAGE_BREAK,
        "```",
        "",
        "# Molecular axes and nematic order",
        "",
        "The chemically directed end-to-end axis points from local atom 18, "
        "the terminal butyl carbon, to local atom 2, the nitrile nitrogen. "
        "The core axis points from local atom 11, the butoxy-side ipso carbon, "
        "to local atom 5, the nitrile-side ipso carbon. The third axis is the "
        "eigenvector of the all-atom mass-weighted inertia tensor having the "
        "smallest principal moment. Cpptraj applies `autoimage` before all axes "
        "are calculated.",
        "",
        "For every frame, all 1,000 molecular unit vectors define",
        "",
        "$$Q = N^{-1}\\sum_i [3u_i u_i^{\\mathsf T}/2-I/2].$$",
        "",
        "The largest eigenvalue is the framewise nematic order `S(t)` and its "
        "eigenvector is the instantaneous director. The values below are means "
        "over 1,000 frames. The displayed standard errors are naive framewise "
        "errors and do not correct for temporal autocorrelation.",
        "",
    ]
    order_rows = []
    for row in order:
        order_rows.append(
            [
                f"{float(row['temperature_K']):.0f}",
                f"{float(row['mean_S_end_to_end']):.4f}",
                f"{float(row['mean_S_core']):.4f}",
                f"{float(row['mean_S_inertia']):.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["T (K)", "end-to-end S", "core S", "inertia S"], order_rows
        )
    )
    lines.extend(
        [
            "",
            "**Figure 1.** Mean framewise nematic order for the chemically "
            "directed end-to-end and biphenyl-core axes and the smallest-moment "
            "inertia axis.",
            "",
            image("reports/4ocb_order/s_vs_temperature_three_axes.png"),
            "",
            "The close agreement between the three definitions shows that the "
            "observed order loss is not caused by a particular endpoint choice.",
            "",
            "```{=openxml}",
            PAGE_BREAK,
            "```",
            "",
            "# Positional structure",
            "",
            "Cpptraj calculated mass-weighted molecular centers after `autoimage` "
            "and normalized the spherical RDF by the mean trajectory cell volume. "
            "The bin width is 0.01 nm and the range is 0--3.5 nm.",
            "",
        ]
    )
    rdf_rows = []
    for row in rdf:
        temperature = int(float(row["temperature_K"]))
        peak_r, peak_g = rdf_values[temperature]
        rdf_rows.append([str(temperature), f"{peak_r:.3f}", f"{peak_g:.3f}"])
    lines.extend(markdown_table(["T (K)", "first peak r (nm)", "peak g(r)"], rdf_rows))
    lines.extend(
        [
            "",
            "**Figure 2.** Molecular COM--COM radial distributions. Each panel "
            "uses all 1,000 frames and 1,000 molecules.",
            "",
            image("reports/4ocb_com_rdf/com_com_rdf_11panel.png"),
            "",
            "The first shell weakens substantially with temperature, but local "
            "positional packing persists after the large decrease in orientational order.",
            "",
            "```{=openxml}",
            PAGE_BREAK,
            "```",
            "",
            "# Director-resolved orientational correlations",
            "",
            "For each frame, the Q tensor is recalculated from the same axis used "
            "for that correlation. With `r_ij` the molecular COM displacement and "
            "`n(t)` the framewise director, the implemented coordinates are",
            "",
            "```text",
            "z       = |r_ij dot n(t)|",
            "r_perp  = sqrt(r_ij^2 - z^2)",
            "P1_ij   = u_i dot u_j",
            "P2_ij   = [3 (u_i dot u_j)^2 - 1] / 2",
            "```",
            "",
            "Parallel histograms use cylinders of radius 1, 2, and 3 nm and are "
            "mirrored to signed z for display. Perpendicular histograms use the "
            "slab `|r_ij dot n(t)| < 0.2 nm`. Python includes 27 periodic replicas. "
            "The figures show ideal-density-normalized P1- or P2-weighted functions "
            "g1 and g2; the stored tables separately retain g0 and the conditional "
            "correlations `C1=g1/g0` and `C2=g2/g0`.",
            "",
            "The inertia axis is excluded from first-rank analysis because its "
            "eigenvector sign is arbitrary.",
            "",
            "**Figure 3.** First-rank correlations parallel to the framewise director.",
            "",
            image("reports/4ocb_director_paircorr/parallel_g1_11panel.png"),
            "",
            "**Figure 4.** First-rank correlations perpendicular to the framewise director.",
            "",
            image("reports/4ocb_director_paircorr/perpendicular_g1_11panel.png"),
            "",
            "The signed short-range features distinguish locally parallel and "
            "antiparallel packing. The long-range values fluctuate around zero, "
            "as expected for a non-polar nematic phase.",
            "",
            "```{=openxml}",
            PAGE_BREAK,
            "```",
            "",
            "**Figure 5.** Second-rank correlations parallel to the framewise director.",
            "",
            image("reports/4ocb_director_paircorr/parallel_g2_11panel.png"),
            "",
            "**Figure 6.** Second-rank correlations perpendicular to the framewise director.",
            "",
            image("reports/4ocb_director_paircorr/perpendicular_g2_11panel.png"),
            "",
            "In the ordered trajectories, the parallel g2 baseline remains nonzero "
            "at long range and is consistent with the squared order scale. The "
            "baseline decreases strongly from 320 through 330 K. The three axis "
            "definitions give the same qualitative behavior.",
            "",
            "```{=openxml}",
            PAGE_BREAK,
            "```",
            "",
            "# Butoxy--aromatic junction conformations",
            "",
            "Two different torsions are required to describe this linkage. They "
            "share atoms but rotate about different central bonds and must not be "
            "interpreted as the same angle.",
            "",
            "## Aliphatic ether torsion",
            "",
            "The ether torsion is local atoms 16-15-28-11, corresponding to "
            "C-C-O-C(ar), with the C15--O28 bond as the central torsional hinge. "
            "Cpptraj calculates the angle after `autoimage`. Histograms use 360 "
            "one-degree bins on `[0,360)` and contain 1,000,000 samples per temperature.",
            "",
            "The PMF is `-RT ln P(chi)` in kcal/mol and is shifted so that its "
            "minimum is zero independently at each temperature.",
            "",
        ]
    )
    torsion_rows = []
    for row in torsion:
        temperature = int(float(row["temperature_K"]))
        mode, left, left_dg, right, right_dg = torsion_values[temperature]
        torsion_rows.append(
            [
                str(temperature),
                f"{mode:.1f}",
                f"{left:.1f}",
                f"{left_dg:.2f}",
                f"{right:.1f}",
                f"{right_dg:.2f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "T (K)",
                "global mode (deg)",
                "left mode (deg)",
                "left dG (kcal/mol)",
                "right mode (deg)",
                "right dG (kcal/mol)",
            ],
            torsion_rows,
        )
    )
    lines.extend(
        [
            "",
            "**Figure 7.** Probability density of the C-C-O-C(ar) ether torsion.",
            "",
            image("reports/4ocb_junction_dihedral/P_chi_CC_O_Car_11panel.png"),
            "",
            "**Figure 8.** Ether-torsion PMFs in kcal/mol.",
            "",
            image("reports/4ocb_junction_dihedral/PMF_chi_CC_O_Car_11panel.png"),
            "",
            "The trans-like basin near 180 degrees is the global minimum across "
            "the complete temperature series. The two side basins are nearly "
            "symmetric and become modestly less costly with increasing temperature.",
            "",
            "```{=openxml}",
            PAGE_BREAK,
            "```",
            "",
            "## Aryl--O ring-junction torsion",
            "",
            "The ring-plane junction torsion is local atoms 10-11-28-15, "
            "C(ring)-C(ipso)-O-C, centered on the aryl--O bond 11--28. Local atom "
            "10 is the adjacent aromatic carbon that establishes the ring reference "
            "plane. Choosing the other adjacent ring carbon, atom 12, would describe "
            "the same rotation with an approximately 180-degree convention shift.",
            "",
        ]
    )
    ring_rows = []
    for row in ring_torsion:
        temperature = int(float(row["temperature_K"]))
        zero, zero_dg, trans, trans_dg, barrier_90, barrier_270 = (
            ring_torsion_values[temperature]
        )
        ring_rows.append(
            [
                str(temperature),
                f"{zero:.1f}",
                f"{zero_dg:.2f}",
                f"{trans:.1f}",
                f"{trans_dg:.2f}",
                f"{barrier_90:.2f}",
                f"{barrier_270:.2f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "T (K)",
                "0-deg mode",
                "dG at 0",
                "180-deg mode",
                "dG at 180",
                "barrier near 90",
                "barrier near 270",
            ],
            ring_rows,
        )
    )
    lines.extend(
        [
            "",
            "**Figure 9.** Probability density of the aryl--O ring-junction torsion.",
            "",
            image("reports/4ocb_ring_junction_dihedral/P_phi_Cring_Cipso_O_C_11panel.png"),
            "",
            "**Figure 10.** Aryl--O ring-junction PMFs in kcal/mol.",
            "",
            image("reports/4ocb_ring_junction_dihedral/PMF_phi_Cring_Cipso_O_C_11panel.png"),
            "",
            "The 0- and 180-degree basins differ by at most about 0.06 kcal/mol "
            "in the present series and are effectively degenerate. They correspond "
            "to the O--alkyl junction plane lying approximately in the aromatic "
            "reference plane. The PMF maxima near 90 and 270 degrees represent "
            "approximately normal junction-plane arrangements. This is the direct "
            "ring-plane comparison with the 5CB C09-C10-C14-C15 torsion; it is not "
            "the same coordinate as the C-C-O-C(ar) ether torsion above.",
            "",
            "# Reproducibility and limitations",
            "",
            "The source trajectories were read from "
            "`/hickory/proj/cdpoonlab/users/emainas/tinker9-4ocb-x1000-pt2.2` "
            "without modification. All cpptraj inputs, logs, raw tables, framewise "
            "arrays, and reductions are under "
            "`data-processed/4ocb/amoeba-pt2.2/`; figures are under `reports/4ocb_*`. "
            "The complete command sequence is documented in `workflows/4ocb/README.md`.",
            "",
            "The report compares independent temperature runs but does not establish "
            "equilibration, statistical independence, phase-transition hysteresis, "
            "or a thermodynamic transition temperature. Framewise error bars do not "
            "replace a block or autocorrelation analysis. Dipole correlations remain "
            "reserved for a separate analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def enforce_font(source: Path, destination: Path, font: str) -> None:
    pattern = re.compile(rb'w:(ascii|hAnsi|eastAsia|cs)="[^"]*"')

    def replace(match: re.Match[bytes]) -> bytes:
        return b"w:" + match.group(1) + b'="' + font.encode() + b'"'

    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as outgoing:
        referenced_media: set[str] = set()
        for name in incoming.namelist():
            if not name.endswith(".rels"):
                continue
            relationships = incoming.read(name).decode("utf-8", errors="ignore")
            for target in re.findall(r'Target="[^"]*media/([^"]+)"', relationships):
                referenced_media.add(f"word/media/{target}")
        for info in incoming.infolist():
            if (
                info.filename.startswith("word/media/")
                and info.filename not in referenced_media
            ):
                continue
            content = incoming.read(info.filename)
            if info.filename in {"word/styles.xml", "word/document.xml"}:
                content = pattern.sub(replace, content)
            outgoing.writestr(info, content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data-processed/4ocb/amoeba-pt2.2"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/4ocb_results_report")
    )
    parser.add_argument(
        "--reference-doc",
        type=Path,
        default=Path("reports/5cb_results_report/5CB_results_report_v2.docx"),
    )
    parser.add_argument("--font", default="Avenir Book")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    reference = (REPO_ROOT / args.reference_doc).resolve()
    require(
        [
            data_root / "order" / "summary.csv",
            data_root / "com-rdf" / "summary.csv",
            data_root / "director-paircorr" / "summary.csv",
            data_root / "junction-dihedral" / "summary.csv",
            data_root / "ring-junction-dihedral" / "summary.csv",
            reference,
        ]
    )
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise SystemExit("pandoc is required to build the DOCX")

    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "4OCB_results_report.md"
    docx_path = out_dir / "4OCB_results_report_v1.docx"
    markdown_path.write_text(build_markdown(data_root))
    with tempfile.TemporaryDirectory(prefix="4ocb-report-") as temporary:
        intermediate = Path(temporary) / "report.docx"
        subprocess.run(
            [
                pandoc,
                str(markdown_path),
                "--from=markdown+raw_attribute+tex_math_dollars",
                f"--reference-doc={reference}",
                f"--resource-path={REPO_ROOT}",
                "--standalone",
                "--output",
                str(intermediate),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        enforce_font(intermediate, docx_path, args.font)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {docx_path}")


if __name__ == "__main__":
    main()
