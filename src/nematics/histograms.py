from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Histogram1D:
    """Fortran-style 1D histogram with per-bin auxiliary channels."""

    min: float
    max: float
    size: int
    ylen: int = 1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.max <= self.min:
            raise ValueError("max must be > min")
        if self.ylen <= 0:
            raise ValueError("ylen must be > 0")

        self.hist = np.zeros(self.size, dtype=float)
        self.histy = np.zeros((self.size, self.ylen), dtype=float, order="F")
        self.dw = self.max - self.min

    def _bin_index(self, x: float) -> int | None:
        if x < self.min or x > self.max:
            return None
        ib = int(self.size * (x - self.min) / self.dw)
        if ib >= self.size:
            ib = self.size - 1
        if ib < 0:
            return None
        return ib

    def update_single(
        self,
        x: float,
        mult: float | None = None,
        y: np.ndarray | list[float] | None = None,
    ) -> None:
        ib = self._bin_index(float(x))
        if ib is None:
            return

        self.hist[ib] += 1.0

        if y is None:
            return

        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if y_arr.size > self.ylen:
            raise ValueError(f"y has length {y_arr.size}, expected <= {self.ylen}")

        if mult is not None:
            self.histy[ib, : y_arr.size] += y_arr / mult
        else:
            self.histy[ib, : y_arr.size] += y_arr

    def normalize(self) -> None:
        ss = float(np.sum(self.hist))
        if ss <= 0.0:
            return

        nz = self.hist > 0.0
        self.histy[nz] /= self.hist[nz, None]
        self.hist = self.size * self.hist / ss

    def centers(self) -> np.ndarray:
        return self.min + self.dw / self.size * (np.arange(self.size, dtype=float) + 0.5)

    def save_to_file(self, path: str | Path, nsamples: float | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        centers = self.centers()
        if nsamples is None:
            hist = self.hist
            histy = self.histy
        else:
            hist = self.hist / nsamples
            histy = self.histy / nsamples

        out = np.column_stack([centers, hist, histy])
        np.savetxt(path, out, fmt="%20.5f")
        return path


@dataclass
class HistogramSet:
    """Grouped 1D histograms matching the Fortran pair-correlation outputs."""

    par_all: Histogram1D
    par_100: Histogram1D
    par_200: Histogram1D
    par_300: Histogram1D
    per: Histogram1D
    rad: Histogram1D


def make_default_paircorr_hist_set() -> HistogramSet:
    kwargs = dict(min=0.0, max=5.0, size=200, ylen=10)
    return HistogramSet(
        par_all=Histogram1D(**kwargs),
        par_100=Histogram1D(**kwargs),
        par_200=Histogram1D(**kwargs),
        par_300=Histogram1D(**kwargs),
        per=Histogram1D(**kwargs),
        rad=Histogram1D(**kwargs),
    )
