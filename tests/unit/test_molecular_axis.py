from __future__ import annotations

from pathlib import Path

import yaml

from nematics import molecular_axis


def _write_config(tmp_path: Path, trajectory: Path, output: Path) -> Path:
    config = {
        "version": 1,
        "system": {
            "molecule": "test",
            "dataset": "amoeba-test",
            "run_id": "run-300",
            "temperature_k": 300,
        },
        "trajectory": {
            "path": str(trajectory),
            "format": "tinker_arc",
            "molecules": 2,
            "atoms_per_molecule": 3,
        },
        "frames": {"start": 1, "stop": 10, "stride": 2},
        "molecular_axis": {
            "name": "end-to-end",
            "start_atom": 1,
            "end_atom": 3,
            "pbc": "minimum_image",
            "normalize": True,
        },
        "output": {"path": str(output), "overwrite": False},
        "execution": {"mode": "slurm"},
        "slurm": {
            "partition": "general",
            "time": "00:10:00",
            "memory": "1G",
            "cpus_per_task": 1,
            "job_name": "axis-test",
            "modules": [],
        },
        "fortran": {
            "compiler": "gfortran",
            "flags": ["-O2", "-std=f2008"],
        },
    }
    path = tmp_path / "axis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_load_molecular_axis_config(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    processed = repo / "data-processed"
    processed.mkdir(parents=True)
    trajectory = tmp_path / "trajectory.arc"
    trajectory.write_text("", encoding="utf-8")
    output = processed / "test" / "axis.bin"
    config_path = _write_config(tmp_path, trajectory, output)
    monkeypatch.setattr(molecular_axis, "_repo_root", lambda: repo)

    config = molecular_axis.load_molecular_axis_config(config_path)

    assert config.trajectory == trajectory
    assert config.output_path == output
    assert config.start_atom == 1
    assert config.end_atom == 3
    assert config.frame_stride == 2
    assert config.use_minimum_image


def test_prepare_writes_namelist_and_slurm_without_submission(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    processed = repo / "data-processed"
    source = repo / "src" / "nematics" / "_molecular_axis.f90"
    source.parent.mkdir(parents=True)
    source.write_text("program placeholder\nend program\n", encoding="utf-8")
    processed.mkdir(parents=True)
    trajectory = tmp_path / "trajectory.arc"
    trajectory.write_text("", encoding="utf-8")
    output = processed / "test" / "axis.bin"
    config_path = _write_config(tmp_path, trajectory, output)
    monkeypatch.setattr(molecular_axis, "_repo_root", lambda: repo)

    job = molecular_axis.prepare_molecular_axis_job(config_path)

    namelist = job.namelist_path.read_text(encoding="utf-8")
    script = job.slurm_script_path.read_text(encoding="utf-8")
    assert "start_atom = 1" in namelist
    assert "end_atom = 3" in namelist
    assert "#SBATCH --partition=general" in script
    assert "srun" in script
    assert "sbatch" not in script
    assert not output.exists()
