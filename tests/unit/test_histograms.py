import numpy as np

from nematics.histograms import Histogram1D


def test_histogram_update_and_save(tmp_path):
    h = Histogram1D(min=0.0, max=1.0, size=10, ylen=3)
    h.update_single(0.2, mult=2.0, y=[1.0, 2.0, 3.0])
    h.update_single(0.2, mult=2.0, y=[1.0, 2.0, 3.0])

    # Two insertions in same bin.
    assert np.isclose(np.sum(h.hist), 2.0)

    out = tmp_path / "h.dat"
    h.save_to_file(out, nsamples=2)

    arr = np.loadtxt(out)
    assert arr.shape == (10, 5)
