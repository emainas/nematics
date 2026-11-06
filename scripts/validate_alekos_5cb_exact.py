#!/usr/bin/env python3
"""Compare accelerated 5CB arrays with Alekos's exact-source text output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


RUNS = (("gro4", 386), ("gro3", 376), ("gro2", 366), ("gro", 356))
OBSERVABLES = (
    ("global_par_1", "5CB_par_1.0"),
    ("global_par_2", "5CB_par_2.0"),
    ("global_par_3", "5CB_par_3.0"),
    ("global_per", "5CB_per"),
    ("cn_rad", "CN_rad"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for run, temperature in RUNS:
        with np.load(args.data_root / run / "correlations.npz") as saved:
            nsnaps = int(saved["nsnaps"])
            for array_name, exact_stem in OBSERVABLES:
                accelerated = np.asarray(saved[array_name], dtype=float) / nsnaps
                raw = np.loadtxt(
                    args.exact_root / run / f"{temperature}_{exact_stem}.dat"
                )
                # Alekos saves: count, 1, dot, dot^2.  The report's g2 is P2.
                exact = np.column_stack(
                    (
                        raw[:, 1],
                        raw[:, 2],
                        raw[:, 3],
                        (3.0 * raw[:, 4] - raw[:, 2]) / 2.0,
                    )
                )
                difference = accelerated - exact
                records.append(
                    {
                        "run": run,
                        "temperature_K": temperature,
                        "observable": array_name,
                        "frames": nsnaps,
                        "max_abs_difference": np.max(np.abs(difference)),
                        "rms_difference": np.sqrt(np.mean(difference**2)),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output}")
    print(f"Largest absolute difference: {max(r['max_abs_difference'] for r in records):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
