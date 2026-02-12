from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fortran_backend import load_paircorr_fortran
from .gro import read_gro_frame
from .histograms import HistogramSet, make_default_paircorr_hist_set
from .paf import compute_global_paf, make_molecular_axes

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _close_images() -> np.ndarray:
    shifts = []
    for k1 in (-1, 0, 1):
        for k2 in (-1, 0, 1):
            for k3 in (-1, 0, 1):
                shifts.append((k1, k2, k3))
    return np.asarray(shifts, dtype=float)


CLOSE_IMAGES = _close_images()


@dataclass
class PairCorrelationResults:
    nsnaps: int
    hist_5cb: HistogramSet
    hist_cn: HistogramSet


def _paircorr_update(
    pos: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    box: np.ndarray,
    glob_dir: np.ndarray,
    h: HistogramSet,
) -> None:
    nmm = pos.shape[0]
    dzz = 0.2

    rho = nmm * (nmm - 1.0) / (box[0] * box[1] * box[2])

    dnz = rho * np.pi * (h.par_all.max - h.par_all.min) / h.par_all.size
    dnr = 0.5 * rho * np.pi * 2.0 * (2.0 * dzz) * (h.per.max - h.per.min) / h.per.size
    dnr2 = 0.5 * rho * np.pi * 4.0 * (h.rad.max - h.rad.min) / h.rad.size

    for i in range(nmm - 1):
        pos_i = pos[i]
        z_i = z[i]
        y_i = y[i]

        for j in range(i + 1, nmm):
            rij0 = pos[j] - pos_i
            computed_ww = False
            ww = None

            for shift in CLOSE_IMAGES:
                rij = rij0 + shift * box
                rijsq = float(np.dot(rij, rij))
                if rijsq > 2.0 * h.par_all.max * h.par_all.max:
                    continue

                rz = float(np.dot(rij, glob_dir))
                rv_sq = max(0.0, rijsq - rz * rz)
                rv = float(np.sqrt(rv_sq))

                if not computed_ww:
                    ww = np.zeros(10, dtype=float)
                    ww[0] = 1.0
                    ww[1] = float(np.dot(z_i, z[j]))
                    ww[2] = ww[1] * ww[1]
                    ww[3] = float(np.dot(z_i, glob_dir) * np.dot(z[j], glob_dir))
                    ww[4] = ww[3] * ww[3]
                    ww[5] = float(np.dot(y_i, y[j]))
                    computed_ww = True

                abs_rz = abs(rz)
                if rv < 1.0:
                    h.par_100.update_single(abs_rz, dnz * 1.0, ww[:6])
                if rv < 2.0:
                    h.par_200.update_single(abs_rz, dnz * 4.0, ww[:6])
                if rv < 3.0:
                    h.par_300.update_single(abs_rz, dnz * 9.0, ww[:6])
                if rv < 5.0:
                    h.par_all.update_single(abs_rz, dnz * 25.0, ww[:6])
                if abs_rz < dzz:
                    h.per.update_single(rv, dnr * rv, ww[:6])

                h.rad.update_single(float(np.sqrt(rijsq)), dnr2 * rijsq, ww[:6])


def _paircorr_update_fortran(
    pos: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    box: np.ndarray,
    glob_dir: np.ndarray,
    h: HistogramSet,
    fort_mod,
) -> None:
    _ = x  # unused, kept for signature parity with Python backend
    fort_mod.paircorr_fort_mod.paircorr_update(
        np.asfortranarray(pos, dtype=np.float64),
        np.asfortranarray(y, dtype=np.float64),
        np.asfortranarray(z, dtype=np.float64),
        np.asfortranarray(box, dtype=np.float64),
        np.asfortranarray(glob_dir, dtype=np.float64),
        np.asfortranarray(CLOSE_IMAGES, dtype=np.float64),
        np.float64(0.2),
        np.float64(h.par_all.min),
        np.float64(h.par_all.max),
        np.float64(h.per.min),
        np.float64(h.per.max),
        np.float64(h.rad.min),
        np.float64(h.rad.max),
        h.par_100.hist,
        h.par_100.histy,
        h.par_200.hist,
        h.par_200.histy,
        h.par_300.hist,
        h.par_300.histy,
        h.par_all.hist,
        h.par_all.histy,
        h.per.hist,
        h.per.histy,
        h.rad.hist,
        h.rad.histy,
    )


def _save_paircorr_set(h: HistogramSet, out_dir: Path, label: str, prefix: str, nsnaps: int) -> list[Path]:
    out_paths = [
        h.par_100.save_to_file(out_dir / f"{label}_{prefix}_par_1.0.dat", nsamples=nsnaps),
        h.par_200.save_to_file(out_dir / f"{label}_{prefix}_par_2.0.dat", nsamples=nsnaps),
        h.par_300.save_to_file(out_dir / f"{label}_{prefix}_par_3.0.dat", nsamples=nsnaps),
        h.par_all.save_to_file(out_dir / f"{label}_{prefix}_par_all.dat", nsamples=nsnaps),
        h.per.save_to_file(out_dir / f"{label}_{prefix}_per.dat", nsamples=nsnaps),
        h.rad.save_to_file(out_dir / f"{label}_{prefix}_rad.dat", nsamples=nsnaps),
    ]
    return out_paths


def save_paircorr_results(results: PairCorrelationResults, out_dir: str | Path, label: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    out_paths.extend(_save_paircorr_set(results.hist_5cb, out_dir, label, "5CB", results.nsnaps))
    out_paths.extend(_save_paircorr_set(results.hist_cn, out_dir, label, "CN", results.nsnaps))
    return out_paths


def compute_paircorr_for_dir(
    gro_dir: str | Path,
    nframes: int = 1000,
    start_frame: int = 0,
    natoms_per_mol: int = 38,
    show_progress: bool = True,
    save_every: int | None = None,
    out_dir: str | Path | None = None,
    label: str | None = None,
    backend: str = "auto",
    auto_build_fortran: bool = False,
) -> PairCorrelationResults:
    """Compute Fortran-style pair-correlation histograms over a trajectory.

    This mirrors the 11-feb-2026 Fortran flow:
    1) Build molecular axes
    2) Build global director from PAF of CN dipole directions (ddir)
    3) Update 5CB and CN pair-correlation histograms for all pairs/images
    """
    gro_dir = Path(gro_dir)
    hist_5cb = make_default_paircorr_hist_set()
    hist_cn = make_default_paircorr_hist_set()
    if backend not in {"auto", "python", "fortran"}:
        raise ValueError("backend must be one of: auto, python, fortran")

    fort_mod = load_paircorr_fortran(auto_build=auto_build_fortran) if backend != "python" else None
    use_fortran = fort_mod is not None and backend in {"auto", "fortran"}
    if backend == "fortran" and fort_mod is None:
        raise RuntimeError(
            "Fortran backend requested but extension is not available. "
            "Run: nematics build-fortran"
        )

    iterator = range(start_frame, start_frame + nframes)
    if show_progress:
        mode = "fortran" if use_fortran else "python"
        iterator = tqdm(iterator, desc=f"paircorr {gro_dir.name} [{mode}]", unit="frame")

    nsnaps = 0
    out_dir_path = Path(out_dir) if out_dir is not None else None
    label_name = label or gro_dir.name.replace(" ", "_")

    for frame_idx in iterator:
        frame = read_gro_frame(gro_dir / f"frame{frame_idx}.gro", natoms_per_mol=natoms_per_mol)
        axes = make_molecular_axes(frame.positions)

        evals, evecs = compute_global_paf(axes.ddir)
        _ = evals
        glob_dir = evecs[:, 0]

        if use_fortran:
            _paircorr_update_fortran(
                axes.molcm, axes.molxax, axes.molyax, axes.molzax, frame.box, glob_dir, hist_5cb, fort_mod
            )
            _paircorr_update_fortran(
                axes.dpos, axes.molxax, axes.molyax, axes.ddir, frame.box, glob_dir, hist_cn, fort_mod
            )
        else:
            _paircorr_update(axes.molcm, axes.molxax, axes.molyax, axes.molzax, frame.box, glob_dir, hist_5cb)
            _paircorr_update(axes.dpos, axes.molxax, axes.molyax, axes.ddir, frame.box, glob_dir, hist_cn)

        nsnaps += 1
        if save_every and out_dir_path is not None and nsnaps % save_every == 0:
            save_paircorr_results(
                PairCorrelationResults(nsnaps=nsnaps, hist_5cb=hist_5cb, hist_cn=hist_cn),
                out_dir=out_dir_path,
                label=label_name,
            )

    return PairCorrelationResults(nsnaps=nsnaps, hist_5cb=hist_5cb, hist_cn=hist_cn)
