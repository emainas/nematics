from __future__ import annotations

from pathlib import Path

import numpy as np

from .gro import read_gro_frame
from .paf import make_molecular_axes

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    def tqdm(iterable, **kwargs):
        return iterable


def _pick_axis(axes, axis: str) -> np.ndarray:
    if axis == "molzax":
        return axes.molzax
    if axis == "ddir":
        return axes.ddir
    raise ValueError(f"Unsupported axis '{axis}'. Expected 'molzax' or 'ddir'.")


def compute_susceptibility(dirs: np.ndarray) -> float:
    """Compute orientational susceptibility from molecular direction vectors.

    Matches the legacy C++ double loop:

        chi = (1/N) * sum_{i,j} 0.5*(3*(u_i.u_j)^2 - 1)

    where N is the number of molecules and u_i are orientation vectors.

    This implementation is algebraically equivalent but faster, using:

        sum_{i,j} (u_i.u_j)^2 = ||U^T U||_F^2

    for U shaped (N, 3).
    """
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must have shape (n, 3)")

    n = dirs.shape[0]
    if n == 0:
        return 0.0

    c = dirs.T @ dirs  # 3x3
    sum_dot_sq = float(np.sum(c * c))
    chi = 1.5 * sum_dot_sq / n - 0.5 * n
    return chi


def compute_susceptibility_timeseries(
    gro_dir: str | Path,
    nframes: int = 1000,
    start_frame: int = 0,
    natoms_per_mol: int = 38,
    axis: str = "molzax",
    show_progress: bool = True,
) -> np.ndarray:
    """Compute susceptibility for each frame in a trajectory directory."""
    gro_dir = Path(gro_dir)

    chi = np.zeros(nframes, dtype=float)
    iterator = range(start_frame, start_frame + nframes)
    if show_progress:
        iterator = tqdm(iterator, desc=f"susc {gro_dir.name}", unit="frame")

    for k, frame_idx in enumerate(iterator):
        frame = read_gro_frame(gro_dir / f"frame{frame_idx}.gro", natoms_per_mol=natoms_per_mol)
        axes = make_molecular_axes(frame.positions)
        dirs = _pick_axis(axes, axis=axis)
        chi[k] = compute_susceptibility(dirs)

    return chi


def save_susceptibility_timeseries(
    chi: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, chi)
    return out_path
