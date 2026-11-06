from pathlib import Path

import numpy as np

from nematics.gro import read_gro_frame
from nematics.paf import compute_global_paf, make_end_to_end_axes, make_molecular_axes


def test_make_end_to_end_axes_with_periodic_boundary():
    positions = np.zeros((2, 3, 3), dtype=float)
    positions[0, 0] = [9.5, 1.0, 1.0]
    positions[0, 2] = [0.5, 1.0, 1.0]
    positions[1, 0] = [2.0, 2.0, 2.0]
    positions[1, 2] = [2.0, 0.0, 2.0]

    axes = make_end_to_end_axes(positions, 0, 2, box=np.array([10.0, 10.0, 10.0]))

    assert np.allclose(axes, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def test_read_and_paf_frame0():
    gro_dir = Path("/hickory/proj/cdpoonlab/users/emainas/5cb-x1000-gaff-88x88x94-zalign/gro")
    frame = read_gro_frame(gro_dir / "frame0.gro", natoms_per_mol=38)

    assert frame.positions.shape == (1000, 38, 3)
    assert frame.box.shape == (3,)

    axes = make_molecular_axes(frame.positions)
    evals, evecs = compute_global_paf(axes.molzax)

    # Eigenvalues should be sorted descending and sum ~ 0 (traceless tensor)
    assert np.all(np.diff(evals) <= 0)
    assert abs(evals.sum()) < 1e-6

    # Eigenvectors should be orthonormal
    should_be_identity = evecs.T @ evecs
    assert np.allclose(should_be_identity, np.eye(3), atol=1e-6)
