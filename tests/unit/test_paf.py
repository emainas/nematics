import numpy as np

from nematics.paf import make_end_to_end_axes


def test_make_end_to_end_axes_with_periodic_boundary():
    positions = np.zeros((2, 3, 3), dtype=float)
    positions[0, 0] = [9.5, 1.0, 1.0]
    positions[0, 2] = [0.5, 1.0, 1.0]
    positions[1, 0] = [2.0, 2.0, 2.0]
    positions[1, 2] = [2.0, 0.0, 2.0]

    axes = make_end_to_end_axes(positions, 0, 2, box=np.array([10.0, 10.0, 10.0]))

    assert np.allclose(axes, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
