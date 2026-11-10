from __future__ import annotations

from pathlib import Path

import yaml

from nematics import dihedral


def _write_config(
    tmp_path: Path, trajectory: Path, cpptraj: Path, output: Path
) -> Path:
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
            "atoms_per_molecule": 5,
        },
        "frames": {"start": 1, "stop": 10, "stride": 2},
        "dihedral": {
            "name": "junction",
            "atoms": [1, 2, 3, 4],
            "pbc": "autoimage",
            "range": "0_360",
        },
        "cpptraj": {"path": str(cpptraj)},
        "output": {"path": str(output), "overwrite": False},
    }
    path = tmp_path / "dihedral.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_prepare_dihedral_job_writes_all_molecular_masks(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    (repo / "data-processed").mkdir(parents=True)
    trajectory = tmp_path / "trajectory.arc"
    trajectory.write_text("trajectory\n", encoding="utf-8")
    cpptraj = tmp_path / "cpptraj"
    cpptraj.write_text("#!/bin/sh\n", encoding="utf-8")
    cpptraj.chmod(0o755)
    output = repo / "data-processed" / "test" / "dihedral.dat"
    config_path = _write_config(tmp_path, trajectory, cpptraj, output)
    monkeypatch.setattr(dihedral, "_repo_root", lambda: repo)

    job = dihedral.prepare_dihedral_job(config_path)
    script = job.cpptraj_input_path.read_text(encoding="utf-8")

    assert f"parm '{trajectory}'" in script
    assert "autoimage" in script
    assert "dihedral mol_0001 @1 @2 @3 @4" in script
    assert "dihedral mol_0002 @6 @7 @8 @9" in script
    assert "range360" in script
    assert not output.exists()


def test_run_dihedral_local_promotes_valid_cpptraj_output(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    (repo / "data-processed").mkdir(parents=True)
    trajectory = tmp_path / "trajectory.arc"
    trajectory.write_text("trajectory\n", encoding="utf-8")
    cpptraj = tmp_path / "cpptraj"
    cpptraj.write_text("#!/bin/sh\n", encoding="utf-8")
    cpptraj.chmod(0o755)
    output = repo / "data-processed" / "test" / "dihedral.dat"
    config_path = _write_config(tmp_path, trajectory, cpptraj, output)
    monkeypatch.setattr(dihedral, "_repo_root", lambda: repo)
    job = dihedral.prepare_dihedral_job(config_path)

    def fake_run(command, **kwargs):
        assert command == [str(cpptraj), "-i", str(job.cpptraj_input_path)]
        job.partial_output_path.write_text(
            "#Frame mol_0001 mol_0002\n1 10.0 20.0\n2 11.0 21.0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(dihedral.subprocess, "run", fake_run)

    result, metadata = dihedral.run_dihedral_local(job)

    assert result == output
    assert result.read_text(encoding="utf-8").startswith("#Frame")
    assert metadata.is_file()
    assert job.performance_path.is_file()
    assert job.reproduce_path.is_file()
    assert not job.partial_output_path.exists()

