import numpy as np
import pytest

from nematics.datasets import get_dataset
from nematics.gro import read_gro_frame
from nematics.paf import compute_global_paf, make_molecular_axes


pytestmark = pytest.mark.integration


def test_read_and_paf_external_gaff_frame():
    frame_path = get_dataset("gaff-5cb").raw_path() / "gro" / "frame0.gro"
    if not frame_path.is_file():
        pytest.skip(f"External trajectory frame is unavailable: {frame_path}")

    frame = read_gro_frame(frame_path, natoms_per_mol=38)

    assert frame.positions.shape == (1000, 38, 3)
    assert frame.box.shape == (3,)

    axes = make_molecular_axes(frame.positions)
    eigenvalues, eigenvectors = compute_global_paf(axes.molzax)

    assert np.all(np.diff(eigenvalues) <= 0)
    assert abs(eigenvalues.sum()) < 1e-6
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3), atol=1e-6)
