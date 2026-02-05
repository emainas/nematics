from pathlib import Path

import numpy as np

from nematics.gro import read_gro_frame
from nematics.paf import compute_global_paf, make_molecular_axes


def test_read_and_paf_frame0():
    gro_dir = Path("data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro")
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
