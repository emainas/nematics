from __future__ import annotations

from pathlib import Path

import numpy as np

from .gro import read_gro_frame
from .paf import compute_global_paf, make_molecular_axes

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    def tqdm(iterable, **kwargs):
        return iterable


def _default_gro_dirs(data_root: Path) -> list[Path]:
    return [
        data_root / "gro",
        data_root / "gro 2",
        data_root / "gro 3",
        data_root / "gro 4",
    ]


def _pick_axis(axes, axis: str) -> np.ndarray:
    if axis == "molzax":
        return axes.molzax
    if axis == "ddir":
        return axes.ddir
    raise ValueError(f"Unsupported axis '{axis}'. Expected 'molzax' or 'ddir'.")


def compute_order_for_dir(
    gro_dir: Path,
    nframes: int = 1000,
    natoms_per_mol: int = 38,
    show_progress: bool = True,
    axis: str = "molzax",
) -> np.ndarray:
    """Compute largest PAF eigenvalue S for each frame in a trajectory."""
    s_values = np.zeros(nframes, dtype=float)
    iterator = range(nframes)
    if show_progress:
        iterator = tqdm(iterator, desc=gro_dir.name, unit="frame")

    for i in iterator:
        frame_path = gro_dir / f"frame{i}.gro"
        frame = read_gro_frame(frame_path, natoms_per_mol=natoms_per_mol)
        axes = make_molecular_axes(frame.positions)
        evals, _ = compute_global_paf(_pick_axis(axes, axis=axis))
        s_values[i] = evals[0]

    return s_values


def compute_and_save_order(
    data_root: str | Path = "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign",
    out_dir: str | Path = "data/order",
    nframes: int = 1000,
    natoms_per_mol: int = 38,
    show_progress: bool = True,
    axis: str = "molzax",
) -> list[Path]:
    """Compute S for each trajectory and save as .npy files.

    Returns list of output file paths.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gro_dirs = _default_gro_dirs(data_root)
    outputs: list[Path] = []

    for gro_dir in gro_dirs:
        s_values = compute_order_for_dir(
            gro_dir,
            nframes=nframes,
            natoms_per_mol=natoms_per_mol,
            show_progress=show_progress,
            axis=axis,
        )
        suffix = "_S.npy" if axis == "molzax" else f"_{axis}_S.npy"
        out_path = out_dir / f"{gro_dir.name.replace(' ', '_')}{suffix}"
        np.save(out_path, s_values)
        outputs.append(out_path)

    return outputs
