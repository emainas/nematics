from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

# Use non-interactive backend for batch report generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEMP_MAP = {
    "gro": 356.0,
    "gro_2": 366.0,
    "gro_3": 376.0,
    "gro_4": 386.0,
}

PAR_KEYS = ["par_1.0", "par_2.0", "par_3.0", "par_all"]


@dataclass(frozen=True)
class PaircorrRecord:
    stem: str
    species: str
    corr: str
    path: Path


def discover_paircorr_files(paircorr_dir: Path) -> list[PaircorrRecord]:
    patt = re.compile(r"^(gro(?:_\d+)?)_(5CB|CN)_(.+)\.dat$")
    records: list[PaircorrRecord] = []
    for path in sorted(paircorr_dir.glob("*.dat")):
        m = patt.match(path.name)
        if not m:
            continue
        stem, species, corr = m.groups()
        records.append(PaircorrRecord(stem=stem, species=species, corr=corr, path=path))
    if not records:
        raise FileNotFoundError(f"No paircorr .dat files found in {paircorr_dir}")
    return records


def load_table(records: list[PaircorrRecord]) -> dict[tuple[str, str, str], np.ndarray]:
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for rec in records:
        out[(rec.stem, rec.species, rec.corr)] = np.loadtxt(rec.path)
    return out


def ordered_stems(records: list[PaircorrRecord]) -> list[str]:
    stems = sorted({rec.stem for rec in records})
    return sorted(stems, key=lambda s: TEMP_MAP.get(s, 1e9))


def stem_label(stem: str) -> str:
    if stem in TEMP_MAP:
        return f"{int(TEMP_MAP[stem])}K"
    return stem


def _style() -> None:
    style_path = Path(__file__).with_name("prl.mplstyle")
    if style_path.exists():
        pass
        #plt.style.use(style_path)


def _plot_par_family(
    data: dict[tuple[str, str, str], np.ndarray],
    stems: list[str],
    species: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    axes = axes.ravel()

    for ax, corr in zip(axes, PAR_KEYS):
        for stem in stems:
            arr = data.get((stem, species, corr))
            if arr is None:
                continue
            x = arr[:, 0]
            g1 = arr[:, 2]  # <u_i dot u_j> channel from Fortran WW(2)
            ax.plot(x, g1, lw=1.4, label=stem_label(stem))

        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(corr.replace("par_", "R<"))
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("g1(z)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"{species} pair correlations along director", y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)


def _plot_perp_radial(
    data: dict[tuple[str, str, str], np.ndarray],
    stems: list[str],
    species: str,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    for stem in stems:
        per = data.get((stem, species, "per"))
        rad = data.get((stem, species, "rad"))
        if per is not None:
            ax1.plot(per[:, 0], per[:, 2], lw=1.4, label=stem_label(stem))
        if rad is not None:
            ax2.plot(rad[:, 0], rad[:, 2], lw=1.4, label=stem_label(stem))

    ax1.axhline(0.0, color="0.7", lw=0.8)
    ax2.axhline(0.0, color="0.7", lw=0.8)
    ax1.set_title(f"{species}: perpendicular (thin-slab)")
    ax2.set_title(f"{species}: radial")
    ax1.set_xlabel("r_perp (nm)")
    ax2.set_xlabel("r (nm)")
    ax1.set_ylabel("g1")
    ax2.set_ylabel("g1")
    ax1.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)


def _plot_cn_dipole_radial_report_style(
    data: dict[tuple[str, str, str], np.ndarray],
    stems: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)

    for stem in stems:
        arr = data.get((stem, "CN", "rad"))
        if arr is None:
            continue
        x = arr[:, 0]
        g1 = arr[:, 2]
        ax.plot(x, g1, lw=1.6, label=stem_label(stem))
        ax.fill_between(x, 0.0, g1, alpha=0.08)

    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlim(0.0, 5.0)
    ax.set_xlabel("inter-dipole distance r (nm)")
    ax.set_ylabel("g1(r)")
    ax.set_title("Dipole-dipole radial correlations")
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot pair-correlation analysis figures from data/paircorr/*.dat"
    )
    parser.add_argument("--paircorr-dir", default="data/paircorr", help="Directory with paircorr .dat files")
    parser.add_argument("--out-dir", default="reports/paircorr", help="Output directory for generated figures")
    args = parser.parse_args()

    paircorr_dir = Path(args.paircorr_dir)
    out_dir = Path(args.out_dir)

    records = discover_paircorr_files(paircorr_dir)
    data = load_table(records)
    stems = ordered_stems(records)

    _style()
    _plot_par_family(data, stems, "5CB", out_dir / "paircorr_5cb_along_director.png")
    _plot_par_family(data, stems, "CN", out_dir / "paircorr_cn_along_director.png")
    _plot_perp_radial(data, stems, "5CB", out_dir / "paircorr_5cb_perp_radial.png")
    _plot_perp_radial(data, stems, "CN", out_dir / "paircorr_cn_perp_radial.png")
    _plot_cn_dipole_radial_report_style(data, stems, out_dir / "paircorr_cn_dipole_radial.png")

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
