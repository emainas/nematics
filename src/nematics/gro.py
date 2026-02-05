from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GroFrame:
    positions: np.ndarray  # (nmols, natoms_per_mol, 3)
    box: np.ndarray  # (3,)
    title: str


def _parse_atom_line(line: str) -> tuple[int, int, np.ndarray]:
    """Return (resnr, atomnr, pos) from a GRO atom line.

    Uses fixed-width columns per GROMACS .gro format, with a fallback split.
    """
    try:
        resnr = int(line[0:5])
        atomnr = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        return resnr, atomnr, np.array([x, y, z], dtype=float)
    except ValueError:
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed GRO atom line: {line!r}")
        resnr = int(parts[0])
        atomnr = int(parts[3])
        x, y, z = map(float, parts[4:7])
        return resnr, atomnr, np.array([x, y, z], dtype=float)


def read_gro_frame(path: str | Path, natoms_per_mol: int = 38) -> GroFrame:
    """Read a single .gro frame and return positions grouped by molecule.

    The Fortran code assumes 1000 molecules and 38 atoms per molecule for 5CB.
    This function defaults to 38 atoms per molecule and infers nmols from the
    atom count. It also uses residue number to place atoms into molecule slots.
    """
    path = Path(path)
    with path.open() as handle:
        title = handle.readline().rstrip("\n")
        natoms = int(handle.readline().strip())

        nmols = natoms // natoms_per_mol
        if nmols * natoms_per_mol != natoms:
            raise ValueError(
                f"Atom count {natoms} not divisible by natoms_per_mol={natoms_per_mol}"
            )

        positions = np.zeros((nmols, natoms_per_mol, 3), dtype=float)

        for _ in range(natoms):
            line = handle.readline()
            resnr, atomnr, pos = _parse_atom_line(line)
            molid = resnr
            local_index = atomnr - (molid - 1) * natoms_per_mol
            if local_index < 1 or local_index > natoms_per_mol:
                raise ValueError(
                    f"Atom index out of range for molecule {molid}: {local_index}"
                )
            positions[molid - 1, local_index - 1, :] = pos

        box_line = handle.readline().split()
        if len(box_line) < 3:
            raise ValueError(f"Malformed box line: {box_line}")
        box = np.array(list(map(float, box_line[:3])), dtype=float)

    return GroFrame(positions=positions, box=box, title=title)


def iter_gro_frames(directory: str | Path, frames: Iterable[int]) -> Iterable[GroFrame]:
    """Yield GroFrame objects from a directory containing frame*.gro files."""
    directory = Path(directory)
    for frame in frames:
        yield read_gro_frame(directory / f"frame{frame}.gro")
