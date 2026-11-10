from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    trajectory = tmp_path / "two.arc"
    trajectory.write_text(
        """     2  two molecules
   100.0 100.0 100.0 90.0 90.0 90.0
     1 C 0.0 0.0 0.0 1
     2 C 1.0 0.0 0.0 1
""",
        encoding="utf-8",
    )

    vectors = tmp_path / "vectors.bin"
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    with vectors.open("wb") as handle:
        handle.write(struct.pack("=8siiiqqii", b"NEMAXIS1", 1, 0x01020304, 4, 1, 2, 3, 44))
        axes.ravel(order="F").tofile(handle)

    director = tmp_path / "director.bin"
    record = np.zeros(12, dtype=np.float64)
    record[:3] = [1.0, -0.5, -0.5]
    record[3:6] = [1.0, 0.0, 0.0]
    with director.open("wb") as handle:
        handle.write(struct.pack("=8siiiqiiii", b"NEMORDR1", 1, 0x01020304, 8, 1, 3, 9, 12, 44))
        record.tofile(handle)
    return trajectory, vectors, director


@pytest.mark.parametrize("rank,expected_sum", [(0, 1.0), (1, 0.0), (2, -0.5)])
def test_corr_rank_weights(tmp_path: Path, rank: int, expected_sum: float) -> None:
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is not available")
    repo = Path(__file__).resolve().parents[2]
    executable = tmp_path / "corr"
    subprocess.run(
        [compiler, "-O2", "-std=f2008", str(repo / "src/nematics/_corr.f90"), "-o", str(executable)],
        check=True,
    )
    trajectory, vectors, director = _write_inputs(tmp_path)
    parallel = tmp_path / f"parallel-{rank}.csv"
    perpendicular = tmp_path / f"perpendicular-{rank}.csv"
    radial = tmp_path / f"radial-{rank}.csv"
    metadata = tmp_path / f"corr-{rank}.yaml"
    performance = tmp_path / f"performance-{rank}.csv"
    vector_line = "" if rank == 0 else f"  vector_path = '{vectors}'\n"
    namelist = tmp_path / f"rank-{rank}.nml"
    namelist.write_text(
        "".join(
            [
                "&corr_job\n",
                f"  trajectory_path = '{trajectory}'\n",
                f"  director_path = '{director}'\n",
                vector_line,
                f"  parallel_path = '{parallel}'\n",
                f"  perpendicular_path = '{perpendicular}'\n",
                f"  radial_path = '{radial}'\n",
                f"  metadata_path = '{metadata}'\n",
                f"  performance_path = '{performance}'\n",
                "  molecules = 2\n",
                "  atoms_per_molecule = 1\n",
                f"  correlation_rank = {rank}\n",
                "  bins = 10\n",
                "  radius_count = 1\n",
                "  cylinder_radii = 1.0\n",
                "  position_atom_count = 1\n",
                "  position_atoms = 1\n",
                "  axial_max_nm = 1.0\n",
                "  perpendicular_max_nm = 1.0\n",
                "  radial_max_nm = 1.0\n",
                "  slab_half_thickness_nm = 0.2\n",
                "/\n",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run([str(executable), str(namelist)], check=True)

    table = np.genfromtxt(parallel, delimiter=",", names=True)
    occupied = table[table["raw_pair_count"] > 0]
    assert occupied.size == 1
    assert occupied["raw_pair_count"] == 1
    np.testing.assert_allclose(occupied["raw_p_rank_sum"], expected_sum, atol=1e-12)
    assert occupied["g0"] > 0.0
    np.testing.assert_allclose(occupied["g_rank"] / occupied["g0"], expected_sum, atol=1e-12)
    assert metadata.is_file()
    assert performance.is_file()
