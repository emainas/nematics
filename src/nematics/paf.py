from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MolecularAxes:
    molcm: np.ndarray  # (nmols, 3)
    molzax: np.ndarray  # (nmols, 3)
    molyax: np.ndarray  # (nmols, 3)
    molxax: np.ndarray  # (nmols, 3)
    dpos: np.ndarray  # (nmols, 3)
    ddir: np.ndarray  # (nmols, 3)


def unitvec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero-length vector encountered in unitvec")
    return v / norm


def make_molecular_axes(positions: np.ndarray) -> MolecularAxes:
    """Replicate MOL_AXES_FRAME_1 for all molecules.

    positions: (nmols, natoms_per_mol, 3)
    Uses 1-based Fortran atom indices:
    - molcm from atoms 8, 10, 12
    - molzax from atom 9 -> 10
    - dpos from atoms 3, 4
    - ddir from atom 4 -> 3
    """
    # Convert to 0-based indices
    idx_cm = np.array([7, 9, 11])
    idx_e1_a = 8
    idx_e1_b = 9
    idx_dpos = np.array([2, 3])
    idx_ddir_a = 3
    idx_ddir_b = 2

    molcm = positions[:, idx_cm, :].mean(axis=1)
    e1 = unitvec(positions[:, idx_e1_a, :] - positions[:, idx_e1_b, :])

    molzax = e1.copy()
    molyax = unitvec(e1)
    molxax = unitvec(e1)

    dpos = positions[:, idx_dpos, :].mean(axis=1)
    ddir = unitvec(positions[:, idx_ddir_a, :] - positions[:, idx_ddir_b, :])

    return MolecularAxes(
        molcm=molcm,
        molzax=molzax,
        molyax=molyax,
        molxax=molxax,
        dpos=dpos,
        ddir=ddir,
    )


def compute_global_paf(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute global PAF eigenvalues/eigenvectors from molecular directions.

    Matches Grobal_PAF in the Fortran code:
    - Build symmetric matrix om from dirs
    - Subtract nn/3 from diagonals
    - Eigen-decompose
    - Scale eigenvalues by 1.5/nn
    Returns (evals_desc, evecs_desc) where evals are descending.
    """
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must have shape (n, 3)")

    nn = dirs.shape[0]
    if nn == 0:
        return np.zeros(3), np.zeros((3, 3))

    om = np.zeros((3, 3), dtype=float)
    om[0, 0] = np.sum(dirs[:, 0] ** 2)
    om[1, 1] = np.sum(dirs[:, 1] ** 2)
    om[2, 2] = np.sum(dirs[:, 2] ** 2)
    om[0, 1] = np.sum(dirs[:, 0] * dirs[:, 1])
    om[0, 2] = np.sum(dirs[:, 0] * dirs[:, 2])
    om[1, 2] = np.sum(dirs[:, 1] * dirs[:, 2])
    om[1, 0] = om[0, 1]
    om[2, 0] = om[0, 2]
    om[2, 1] = om[1, 2]

    om[0, 0] -= nn / 3.0
    om[1, 1] -= nn / 3.0
    om[2, 2] -= nn / 3.0

    evals, evecs = np.linalg.eigh(om)
    # sort descending to match Fortran usage
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evals = 1.5 * evals / nn
    return evals, evecs
