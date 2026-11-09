from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_fortran_nematic_order_known_frames(tmp_path: Path) -> None:
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is not available")

    repo = Path(__file__).resolve().parents[2]
    source = repo / "src" / "nematics" / "_nematic_order.f90"
    executable = tmp_path / "nematic-order"
    subprocess.run(
        [compiler, "-O2", "-std=f2008", str(source), "-o", str(executable)],
        check=True,
    )

    input_path = tmp_path / "molecular-axis.bin"
    axes = np.zeros((2, 4, 3), dtype=np.float32)
    axes[0, :, 2] = 1.0
    axes[1, :3, 0] = 1.0
    axes[1, 3, 1] = 1.0
    with input_path.open("wb") as handle:
        handle.write(
            struct.pack(
                "=8siiiqqii",
                b"NEMAXIS1",
                1,
                0x01020304,
                4,
                2,
                4,
                3,
                44,
            )
        )
        axes.tofile(handle)

    output_path = tmp_path / "nematic-order.bin.partial"
    csv_path = tmp_path / "nematic-order.csv.partial"
    metadata_path = tmp_path / "nematic-order.meta.yaml.partial"
    performance_path = tmp_path / "performance.csv.partial"
    namelist = tmp_path / "job.nml"
    namelist.write_text(
        "\n".join(
            [
                "&nematic_order_job",
                f"  input_path = '{input_path}'",
                f"  output_path = '{output_path}'",
                f"  final_output_path = '{tmp_path / 'nematic-order.bin'}'",
                f"  csv_path = '{csv_path}'",
                f"  metadata_path = '{metadata_path}'",
                f"  performance_path = '{performance_path}'",
                "  molecule_name = 'test'",
                "  dataset_name = 'synthetic'",
                "  run_id = 'known-frames'",
                "  temperature_k = 300.0",
                "/",
                "",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run([str(executable), str(namelist)], check=True)

    with output_path.open("rb") as handle:
        header = struct.unpack("=8siiiqiiii", handle.read(44))
        records = np.fromfile(handle, dtype=np.float64).reshape(2, 12)
    assert header == (b"NEMORDR1", 1, 0x01020304, 8, 2, 3, 9, 12, 44)
    np.testing.assert_allclose(records[0, :3], [1.0, -0.5, -0.5], atol=1e-12)
    np.testing.assert_allclose(records[0, 3:6], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(records[1, :3], [0.625, -0.125, -0.5], atol=1e-12)
    np.testing.assert_allclose(records[1, 3:6], [1.0, 0.0, 0.0], atol=1e-12)
    assert csv_path.is_file()
    assert metadata_path.is_file()
    assert performance_path.is_file()
