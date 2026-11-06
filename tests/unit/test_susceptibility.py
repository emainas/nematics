import numpy as np

from nematics.susceptibility import compute_susceptibility


def test_susceptibility_all_aligned_is_n():
    dirs = np.tile(np.array([[0.0, 0.0, 1.0]]), (5, 1))
    chi = compute_susceptibility(dirs)
    assert np.isclose(chi, 5.0)


def test_susceptibility_two_orthogonal_unit_vectors():
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # P2 values: [[1, -0.5], [-0.5, 1]], sum=1, chi=sum/N=0.5
    chi = compute_susceptibility(dirs)
    assert np.isclose(chi, 0.5)


def test_susceptibility_empty_returns_zero():
    dirs = np.zeros((0, 3))
    chi = compute_susceptibility(dirs)
    assert chi == 0.0
