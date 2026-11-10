from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DihedralConfig:
    config_path: Path
    repo_root: Path
    molecule: str
    dataset: str
    run_id: str
    temperature_k: float
    trajectory: Path
    topology: Path
    trajectory_format: str
    molecules: int
    atoms_per_molecule: int
    frame_start: int
    frame_stop: int
    frame_stride: int
    name: str
    atoms: tuple[int, int, int, int]
    autoimage: bool
    range_360: bool
    cpptraj_path: Path
    output_path: Path
    overwrite: bool


@dataclass(frozen=True)
class PreparedDihedralJob:
    config: DihedralConfig
    cpptraj_input_path: Path
    cpptraj_log_path: Path
    normalized_config_path: Path
    metadata_path: Path
    reproduce_path: Path
    performance_path: Path
    partial_output_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _required(mapping: dict[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required YAML field: {field}.{key}")
    return mapping[key]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _ensure_under(path: Path, parent: Path, field: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{field} must be under {parent}; got {path}") from exc


def _safe_name(value: Any) -> str:
    name = str(value)
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name):
        raise ValueError("dihedral.name may contain only lowercase letters, digits, '-' and '_'")
    return name


def load_dihedral_config(path: str | Path) -> DihedralConfig:
    """Read and validate one cpptraj dihedral YAML file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "root")

    version = int(root.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported dihedral YAML version: {version}")

    system = _mapping(_required(root, "system", "root"), "system")
    trajectory_cfg = _mapping(_required(root, "trajectory", "root"), "trajectory")
    frames = _mapping(_required(root, "frames", "root"), "frames")
    dihedral = _mapping(_required(root, "dihedral", "root"), "dihedral")
    cpptraj = _mapping(_required(root, "cpptraj", "root"), "cpptraj")
    output = _mapping(_required(root, "output", "root"), "output")

    trajectory = _resolve_path(
        _required(trajectory_cfg, "path", "trajectory"), config_path.parent
    )
    if not trajectory.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {trajectory}")
    topology = _resolve_path(trajectory_cfg.get("topology", trajectory), config_path.parent)
    if not topology.is_file():
        raise FileNotFoundError(f"Topology does not exist: {topology}")

    trajectory_format = str(trajectory_cfg.get("format", "tinker_arc")).lower()
    if trajectory_format != "tinker_arc":
        raise ValueError("Only trajectory.format: tinker_arc is currently supported")
    molecules = _positive_int(
        _required(trajectory_cfg, "molecules", "trajectory"), "trajectory.molecules"
    )
    atoms_per_molecule = _positive_int(
        _required(trajectory_cfg, "atoms_per_molecule", "trajectory"),
        "trajectory.atoms_per_molecule",
    )

    frame_start = _positive_int(frames.get("start", 1), "frames.start")
    frame_stop = int(_required(frames, "stop", "frames"))
    if frame_stop < 0 or (frame_stop and frame_stop < frame_start):
        raise ValueError("frames.stop must be 0 (all) or >= frames.start")
    frame_stride = _positive_int(frames.get("stride", 1), "frames.stride")

    atoms_raw = _required(dihedral, "atoms", "dihedral")
    if not isinstance(atoms_raw, list) or len(atoms_raw) != 4:
        raise ValueError("dihedral.atoms must contain exactly four local atom indices")
    atoms = tuple(_positive_int(value, "dihedral.atoms") for value in atoms_raw)
    if any(atom > atoms_per_molecule for atom in atoms):
        raise ValueError("dihedral.atoms must not exceed atoms_per_molecule")
    if len(set(atoms)) != 4:
        raise ValueError("dihedral.atoms must contain four distinct indices")

    pbc = str(dihedral.get("pbc", "autoimage")).lower()
    if pbc not in {"autoimage", "none"}:
        raise ValueError("dihedral.pbc must be 'autoimage' or 'none'")
    angle_range = str(dihedral.get("range", "0_360")).lower()
    if angle_range not in {"0_360", "-180_180"}:
        raise ValueError("dihedral.range must be '0_360' or '-180_180'")

    cpptraj_path = _resolve_path(_required(cpptraj, "path", "cpptraj"), config_path.parent)
    if not cpptraj_path.is_file() or not os.access(cpptraj_path, os.X_OK):
        raise FileNotFoundError(f"cpptraj executable is unavailable: {cpptraj_path}")

    output_path = _resolve_path(_required(output, "path", "output"), _repo_root())
    _ensure_under(output_path, (_repo_root() / "data-processed").resolve(), "output.path")
    if output_path.suffix != ".dat":
        raise ValueError("output.path must end in .dat")

    return DihedralConfig(
        config_path=config_path,
        repo_root=_repo_root(),
        molecule=str(_required(system, "molecule", "system")),
        dataset=str(_required(system, "dataset", "system")),
        run_id=str(_required(system, "run_id", "system")),
        temperature_k=float(_required(system, "temperature_k", "system")),
        trajectory=trajectory,
        topology=topology,
        trajectory_format=trajectory_format,
        molecules=molecules,
        atoms_per_molecule=atoms_per_molecule,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_stride=frame_stride,
        name=_safe_name(_required(dihedral, "name", "dihedral")),
        atoms=atoms,  # type: ignore[arg-type]
        autoimage=pbc == "autoimage",
        range_360=angle_range == "0_360",
        cpptraj_path=cpptraj_path,
        output_path=output_path,
        overwrite=_boolean(output.get("overwrite", False), "output.overwrite"),
    )


def _cpptraj_quote(path: Path) -> str:
    value = str(path)
    if "\n" in value or "\r" in value or "'" in value:
        raise ValueError(f"Unsupported character in cpptraj path: {value!r}")
    return f"'{value}'"


def _script_text(config: DihedralConfig, partial_output: Path) -> str:
    stop = "last" if config.frame_stop == 0 else str(config.frame_stop)
    lines = [
        f"parm {_cpptraj_quote(config.topology)}",
        (
            f"trajin {_cpptraj_quote(config.trajectory)} {config.frame_start} "
            f"{stop} {config.frame_stride}"
        ),
    ]
    if config.autoimage:
        lines.append("autoimage")
    range_option = " range360" if config.range_360 else ""
    for molecule_index in range(config.molecules):
        offset = molecule_index * config.atoms_per_molecule
        global_atoms = tuple(offset + atom for atom in config.atoms)
        masks = " ".join(f"@{atom}" for atom in global_atoms)
        lines.append(
            f"dihedral mol_{molecule_index + 1:04d} {masks} "
            f"out {_cpptraj_quote(partial_output)}{range_option}"
        )
    lines.extend(("run", "quit", ""))
    return "\n".join(lines)


def _normalized_config(config: DihedralConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "system": {
            "molecule": config.molecule,
            "dataset": config.dataset,
            "run_id": config.run_id,
            "temperature_k": config.temperature_k,
        },
        "trajectory": {
            "path": str(config.trajectory),
            "topology": str(config.topology),
            "format": config.trajectory_format,
            "molecules": config.molecules,
            "atoms_per_molecule": config.atoms_per_molecule,
        },
        "frames": {
            "start": config.frame_start,
            "stop": config.frame_stop,
            "stride": config.frame_stride,
        },
        "dihedral": {
            "name": config.name,
            "atoms": list(config.atoms),
            "pbc": "autoimage" if config.autoimage else "none",
            "range": "0_360" if config.range_360 else "-180_180",
        },
        "cpptraj": {"path": str(config.cpptraj_path)},
        "output": {"path": str(config.output_path), "overwrite": config.overwrite},
    }


def prepare_dihedral_job(path: str | Path) -> PreparedDihedralJob:
    config = load_dihedral_config(path)
    output_dir = config.output_path.parent
    provenance_dir = output_dir / "provenance"
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    partial_output = config.output_path.with_name(f"{config.output_path.name}.partial")
    job = PreparedDihedralJob(
        config=config,
        cpptraj_input_path=provenance_dir / "cpptraj.in",
        cpptraj_log_path=provenance_dir / "cpptraj.log",
        normalized_config_path=provenance_dir / "config.yaml",
        metadata_path=output_dir / "dihedral.meta.yaml",
        reproduce_path=output_dir / "reproduce.yaml",
        performance_path=output_dir / "performance.csv",
        partial_output_path=partial_output,
    )
    job.cpptraj_input_path.write_text(_script_text(config, partial_output), encoding="utf-8")
    job.normalized_config_path.write_text(
        yaml.safe_dump(_normalized_config(config), sort_keys=False), encoding="utf-8"
    )
    return job


def _validate_output(job: PreparedDihedralJob) -> int:
    path = job.partial_output_path
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cpptraj did not create a non-empty output: {path}")
    frame_count = 0
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header or header[0] != "#Frame":
            raise RuntimeError(f"Unexpected cpptraj output header in {path}")
        if len(header) != job.config.molecules + 1:
            raise RuntimeError(
                f"Expected {job.config.molecules} angle columns; found {len(header) - 1}"
            )
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                frame_count += 1
    if frame_count == 0:
        raise RuntimeError(f"cpptraj output contains no frames: {path}")
    return frame_count


def run_dihedral_local(job: PreparedDihedralJob) -> tuple[Path, Path]:
    config = job.config
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output exists and output.overwrite is false: {config.output_path}"
        )
    if job.partial_output_path.exists():
        job.partial_output_path.unlink()

    started = time.perf_counter()
    with job.cpptraj_log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(config.cpptraj_path), "-i", str(job.cpptraj_input_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=config.repo_root,
        )
    elapsed = time.perf_counter() - started
    frames = _validate_output(job)

    if config.output_path.exists():
        config.output_path.unlink()
    job.partial_output_path.replace(config.output_path)

    metadata = {
        "format": "nematics-dihedral-v1",
        "source": {
            "trajectory": str(config.trajectory),
            "topology": str(config.topology),
            "molecule": config.molecule,
            "dataset": config.dataset,
            "run_id": config.run_id,
            "temperature_k": config.temperature_k,
        },
        "shape": {"frames": frames, "molecules_per_frame": config.molecules},
        "calculation": {
            "name": config.name,
            "local_atoms_one_based": list(config.atoms),
            "global_atom_rule": "global=(molecule-1)*atoms_per_molecule+local",
            "pbc": "cpptraj autoimage" if config.autoimage else "none",
            "angle_units": "degrees",
            "angle_range": (
                "0 to 360 (periodic; printed 360 is equivalent to 0)"
                if config.range_360
                else "-180 to 180"
            ),
            "frame_start": config.frame_start,
            "frame_stop": config.frame_stop,
            "frame_stride": config.frame_stride,
        },
        "output": str(config.output_path),
        "status": "complete",
    }
    job.metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    reproduce = {
        "command": f"nematics dihedral {config.config_path} --local",
        "config": str(config.config_path),
        "cpptraj": str(config.cpptraj_path),
        "cpptraj_input": str(job.cpptraj_input_path),
    }
    job.reproduce_path.write_text(yaml.safe_dump(reproduce, sort_keys=False), encoding="utf-8")
    with job.performance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "wall_seconds", "frames", "molecules"))
        writer.writerow(("cpptraj", f"{elapsed:.6f}", frames, config.molecules))
    return config.output_path, job.metadata_path

