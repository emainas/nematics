from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .molecular_axis import (
    _boolean,
    _ensure_under,
    _fortran_string,
    _mapping,
    _positive_int,
    _repo_root,
    _required,
    _resolve_path,
    _safe_slurm_value,
)


@dataclass(frozen=True)
class CorrConfig:
    config_path: Path
    repo_root: Path
    molecule: str
    dataset: str
    run_id: str
    temperature_k: float
    trajectory_path: Path
    molecules: int
    atoms_per_molecule: int
    frame_start: int
    frame_stop: int
    frame_stride: int
    director_path: Path
    vector_path: Path | None
    position_atoms: tuple[int, ...]
    rank: int
    all_ranks: bool
    cylinder_radii_nm: tuple[float, ...]
    axial_max_nm: float
    perpendicular_max_nm: float
    radial_max_nm: float
    slab_half_thickness_nm: float
    bins: int
    output_dir: Path
    overwrite: bool
    slurm_partition: str
    slurm_account: str | None
    slurm_qos: str | None
    slurm_time: str
    slurm_memory: str
    slurm_cpus_per_task: int
    slurm_job_name: str
    slurm_modules: tuple[str, ...]
    compiler: str
    compiler_flags: tuple[str, ...]


@dataclass(frozen=True)
class PreparedCorrJob:
    config: CorrConfig
    provenance_dir: Path
    namelist_path: Path
    slurm_script_path: Path
    normalized_config_path: Path
    parallel_path: Path
    perpendicular_path: Path
    radial_path: Path
    metadata_path: Path
    performance_path: Path
    reproduce_path: Path
    partial_parallel_path: Path
    partial_perpendicular_path: Path
    partial_radial_path: Path
    partial_metadata_path: Path
    partial_performance_path: Path
    partial_reproduce_path: Path


def _positive_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def load_corr_config(path: str | Path) -> CorrConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle), "root")
    if int(root.get("version", 1)) != 1:
        raise ValueError("Unsupported corr YAML version")

    system = _mapping(_required(root, "system", "root"), "system")
    trajectory = _mapping(_required(root, "trajectory", "root"), "trajectory")
    frames = _mapping(_required(root, "frames", "root"), "frames")
    inputs = _mapping(_required(root, "inputs", "root"), "inputs")
    analysis = _mapping(_required(root, "analysis", "root"), "analysis")
    position = _mapping(_required(root, "position", "root"), "position")
    output = _mapping(_required(root, "output", "root"), "output")
    slurm = _mapping(_required(root, "slurm", "root"), "slurm")
    fortran = _mapping(root.get("fortran", {}), "fortran")

    repo = _repo_root()
    processed_root = (repo / "data-processed").resolve()
    trajectory_path = _resolve_path(_required(trajectory, "path", "trajectory"), repo)
    director_path = _resolve_path(_required(inputs, "director", "inputs"), repo)
    output_dir = _resolve_path(_required(output, "path", "output"), repo)
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {trajectory_path}")
    if str(trajectory.get("format", "tinker_arc")).lower() != "tinker_arc":
        raise ValueError("Only trajectory.format: tinker_arc is supported")
    _ensure_under(director_path, processed_root, "inputs.director")
    _ensure_under(output_dir, processed_root, "output.path")
    if not director_path.is_file():
        raise FileNotFoundError(f"Director file does not exist: {director_path}")

    ranks_value = analysis.get("ranks")
    if ranks_value is None:
        rank = int(_required(analysis, "rank", "analysis"))
        if rank not in {0, 1, 2}:
            raise ValueError("analysis.rank must be 0, 1, or 2")
        all_ranks = False
    else:
        if "rank" in analysis:
            raise ValueError("analysis must contain either rank or ranks, not both")
        if not isinstance(ranks_value, list):
            raise ValueError("analysis.ranks must be the YAML list [0, 1, 2]")
        try:
            ranks = tuple(int(value) for value in ranks_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis.ranks must be the YAML list [0, 1, 2]") from exc
        if ranks != (0, 1, 2):
            raise ValueError("analysis.ranks currently supports exactly [0, 1, 2]")
        rank = 2
        all_ranks = True
    if int(analysis.get("periodic_images", 27)) != 27:
        raise ValueError("analysis.periodic_images must be 27 for Alekos-compatible corr")
    vector_value = inputs.get("vectors")
    vector_path = _resolve_path(vector_value, repo) if vector_value else None
    if rank > 0 and vector_path is None:
        raise ValueError("inputs.vectors is required for rank 1 or 2")
    if vector_path is not None:
        _ensure_under(vector_path, processed_root, "inputs.vectors")
        if not vector_path.is_file():
            raise FileNotFoundError(f"Vector file does not exist: {vector_path}")

    if str(position.get("definition", "")).lower() != "atom_average":
        raise ValueError("position.definition must be atom_average")
    position_atoms_raw = position.get("local_atoms")
    if not isinstance(position_atoms_raw, list) or not position_atoms_raw:
        raise ValueError("position.local_atoms must be a nonempty YAML list")
    position_atoms = tuple(
        _positive_int(value, "position.local_atoms") for value in position_atoms_raw
    )
    if len(position_atoms) > 32 or len(set(position_atoms)) != len(position_atoms):
        raise ValueError("position.local_atoms must be unique and contain at most 32 atoms")
    configured_atoms = _positive_int(
        _required(trajectory, "atoms_per_molecule", "trajectory"),
        "trajectory.atoms_per_molecule",
    )
    if max(position_atoms) > configured_atoms:
        raise ValueError("position.local_atoms contains an atom outside the molecule")

    radii_raw = analysis.get("cylinder_radii_nm", [1.0, 2.0, 3.0])
    if not isinstance(radii_raw, list) or not radii_raw:
        raise ValueError("analysis.cylinder_radii_nm must be a nonempty YAML list")
    radii = tuple(_positive_float(value, "analysis.cylinder_radii_nm") for value in radii_raw)
    if len(radii) > 32 or len(set(radii)) != len(radii):
        raise ValueError("Cylinder radii must be unique and contain at most 32 values")

    frame_start = _positive_int(frames.get("start", 1), "frames.start")
    frame_stop = int(frames.get("stop", 0))
    frame_stride = _positive_int(frames.get("stride", 1), "frames.stride")
    if frame_stop < 0 or (frame_stop and frame_stop < frame_start):
        raise ValueError("frames.stop must be 0 or not less than frames.start")

    modules_raw = slurm.get("modules", [])
    if not isinstance(modules_raw, list) or not all(
        isinstance(item, str) and item for item in modules_raw
    ):
        raise ValueError("slurm.modules must be a YAML list of module names")
    flags_raw = fortran.get("flags", ["-O3", "-std=f2008"])
    if not isinstance(flags_raw, list) or not all(
        isinstance(item, str) and item for item in flags_raw
    ):
        raise ValueError("fortran.flags must be a YAML list of compiler arguments")
    cpus = _positive_int(slurm.get("cpus_per_task", 1), "slurm.cpus_per_task")
    if cpus != 1:
        raise ValueError("corr is currently serial; slurm.cpus_per_task must be 1")
    account = slurm.get("account")
    qos = slurm.get("qos")

    return CorrConfig(
        config_path=config_path,
        repo_root=repo,
        molecule=str(_required(system, "molecule", "system")),
        dataset=str(_required(system, "dataset", "system")),
        run_id=str(_required(system, "run_id", "system")),
        temperature_k=float(_required(system, "temperature_k", "system")),
        trajectory_path=trajectory_path,
        molecules=_positive_int(_required(trajectory, "molecules", "trajectory"), "trajectory.molecules"),
        atoms_per_molecule=configured_atoms,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_stride=frame_stride,
        director_path=director_path,
        vector_path=vector_path,
        position_atoms=position_atoms,
        rank=rank,
        all_ranks=all_ranks,
        cylinder_radii_nm=radii,
        axial_max_nm=_positive_float(analysis.get("axial_max_nm", 5.0), "analysis.axial_max_nm"),
        perpendicular_max_nm=_positive_float(
            analysis.get("perpendicular_max_nm", 5.0),
            "analysis.perpendicular_max_nm",
        ),
        radial_max_nm=_positive_float(analysis.get("radial_max_nm", 5.0), "analysis.radial_max_nm"),
        slab_half_thickness_nm=_positive_float(
            analysis.get("slab_half_thickness_nm", 0.2),
            "analysis.slab_half_thickness_nm",
        ),
        bins=_positive_int(analysis.get("bins", 200), "analysis.bins"),
        output_dir=output_dir,
        overwrite=_boolean(output.get("overwrite", False), "output.overwrite"),
        slurm_partition=_safe_slurm_value(slurm.get("partition", "general"), "slurm.partition"),
        slurm_account=_safe_slurm_value(account, "slurm.account") if account else None,
        slurm_qos=_safe_slurm_value(qos, "slurm.qos") if qos else None,
        slurm_time=_safe_slurm_value(slurm.get("time", "04:00:00"), "slurm.time"),
        slurm_memory=_safe_slurm_value(slurm.get("memory", "1G"), "slurm.memory"),
        slurm_cpus_per_task=cpus,
        slurm_job_name=_safe_slurm_value(
            slurm.get(
                "job_name",
                f"corr{'012' if all_ranks else rank}-{system.get('molecule', 'system')}",
            ),
            "slurm.job_name",
        ),
        slurm_modules=tuple(modules_raw),
        compiler=str(fortran.get("compiler", "gfortran")),
        compiler_flags=tuple(flags_raw),
    )


def _source_path(config: CorrConfig) -> Path:
    path = config.repo_root / "src" / "nematics" / "_corr.f90"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _namelist_text(job: PreparedCorrJob) -> str:
    config = job.config
    radii = ", ".join(f"{radius:.12g}" for radius in config.cylinder_radii_nm)
    position_atoms = ", ".join(str(atom) for atom in config.position_atoms)
    vector_path = "" if config.vector_path is None else str(config.vector_path)
    return "\n".join(
        [
            "&corr_job",
            f"  trajectory_path = {_fortran_string(config.trajectory_path)}",
            f"  director_path = {_fortran_string(config.director_path)}",
            f"  vector_path = {_fortran_string(vector_path)}",
            f"  parallel_path = {_fortran_string(job.partial_parallel_path)}",
            f"  perpendicular_path = {_fortran_string(job.partial_perpendicular_path)}",
            f"  radial_path = {_fortran_string(job.partial_radial_path)}",
            f"  metadata_path = {_fortran_string(job.partial_metadata_path)}",
            f"  performance_path = {_fortran_string(job.partial_performance_path)}",
            f"  molecule_name = {_fortran_string(config.molecule)}",
            f"  dataset_name = {_fortran_string(config.dataset)}",
            f"  run_id = {_fortran_string(config.run_id)}",
            f"  temperature_k = {config.temperature_k:.12g}",
            f"  molecules = {config.molecules}",
            f"  atoms_per_molecule = {config.atoms_per_molecule}",
            f"  frame_start = {config.frame_start}",
            f"  frame_stop = {config.frame_stop}",
            f"  frame_stride = {config.frame_stride}",
            f"  correlation_rank = {config.rank}",
            f"  all_ranks = {'.true.' if config.all_ranks else '.false.'}",
            f"  bins = {config.bins}",
            f"  radius_count = {len(config.cylinder_radii_nm)}",
            f"  cylinder_radii = {radii}",
            f"  position_atom_count = {len(config.position_atoms)}",
            f"  position_atoms = {position_atoms}",
            f"  axial_max_nm = {config.axial_max_nm:.12g}",
            f"  perpendicular_max_nm = {config.perpendicular_max_nm:.12g}",
            f"  radial_max_nm = {config.radial_max_nm:.12g}",
            f"  slab_half_thickness_nm = {config.slab_half_thickness_nm:.12g}",
            "/",
            "",
        ]
    )


def _all_final_paths(job: PreparedCorrJob) -> tuple[Path, ...]:
    return (
        job.parallel_path,
        job.perpendicular_path,
        job.radial_path,
        job.metadata_path,
        job.performance_path,
        job.reproduce_path,
    )


def _all_partial_paths(job: PreparedCorrJob) -> tuple[Path, ...]:
    return (
        job.partial_parallel_path,
        job.partial_perpendicular_path,
        job.partial_radial_path,
        job.partial_metadata_path,
        job.partial_performance_path,
        job.partial_reproduce_path,
    )


def _slurm_text(job: PreparedCorrJob) -> str:
    config = job.config
    log_path = job.provenance_dir / "slurm-%j.out"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={config.slurm_job_name}",
        f"#SBATCH --partition={config.slurm_partition}",
        f"#SBATCH --time={config.slurm_time}",
        f"#SBATCH --mem={config.slurm_memory}",
        f"#SBATCH --cpus-per-task={config.slurm_cpus_per_task}",
        f"#SBATCH --output={log_path}",
    ]
    if config.slurm_account:
        lines.append(f"#SBATCH --account={config.slurm_account}")
    if config.slurm_qos:
        lines.append(f"#SBATCH --qos={config.slurm_qos}")
    lines.extend(["", "set -euo pipefail", ""])
    if config.slurm_modules:
        lines.append("module purge")
        for module in config.slurm_modules:
            lines.append(f"module load {shlex.quote(module)}")
        lines.append("")
    if not config.overwrite:
        tests = " || ".join(f"-e {shlex.quote(str(path))}" for path in _all_final_paths(job))
        lines.extend(
            [
                f"if [[ {tests} ]]; then",
                "  echo 'Refusing to replace an existing completed output.' >&2",
                "  exit 3",
                "fi",
                "",
            ]
        )
    compiler = shlex.quote(config.compiler)
    flags = " ".join(shlex.quote(flag) for flag in config.compiler_flags)
    runtime_partials = _all_partial_paths(job)[:-1]
    lines.extend(
        [
            "work_root=${SLURM_TMPDIR:-/tmp}",
            'exe="$work_root/nematics-corr-${SLURM_JOB_ID}"',
            f"{compiler} {flags} {shlex.quote(str(_source_path(config)))} -o \"$exe\"",
            "rm -f -- " + " ".join(shlex.quote(str(path)) for path in runtime_partials),
            f"srun \"$exe\" {shlex.quote(str(job.namelist_path))}",
        ]
    )
    move_flag = "-f" if config.overwrite else ""
    for partial, final in zip(_all_partial_paths(job), _all_final_paths(job)):
        command = f"mv {move_flag} -- {shlex.quote(str(partial))} {shlex.quote(str(final))}"
        lines.append(command.replace("mv  --", "mv --"))
    lines.extend(["echo 'Corr calculation completed.'", ""])
    return "\n".join(lines)


def prepare_corr_job(config_or_path: CorrConfig | str | Path) -> PreparedCorrJob:
    config = config_or_path if isinstance(config_or_path, CorrConfig) else load_corr_config(config_or_path)
    output_dir = config.output_dir
    provenance = output_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    job = PreparedCorrJob(
        config=config,
        provenance_dir=provenance,
        namelist_path=provenance / "corr.nml",
        slurm_script_path=provenance / "corr.sbatch",
        normalized_config_path=provenance / "config.yaml",
        parallel_path=output_dir / "parallel.csv",
        perpendicular_path=output_dir / "perpendicular.csv",
        radial_path=output_dir / "radial.csv",
        metadata_path=output_dir / "corr.meta.yaml",
        performance_path=output_dir / "performance.csv",
        reproduce_path=output_dir / "reproduce.yaml",
        partial_parallel_path=output_dir / "parallel.csv.partial",
        partial_perpendicular_path=output_dir / "perpendicular.csv.partial",
        partial_radial_path=output_dir / "radial.csv.partial",
        partial_metadata_path=output_dir / "corr.meta.yaml.partial",
        partial_performance_path=output_dir / "performance.csv.partial",
        partial_reproduce_path=output_dir / "reproduce.yaml.partial",
    )
    existing = [path for path in _all_final_paths(job) if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("Refusing to replace existing output: " + ", ".join(map(str, existing)))
    job.namelist_path.write_text(_namelist_text(job), encoding="utf-8")
    job.slurm_script_path.write_text(_slurm_text(job), encoding="utf-8")
    job.normalized_config_path.write_text(
        yaml.safe_dump(yaml.safe_load(config.config_path.read_text(encoding="utf-8")), sort_keys=False),
        encoding="utf-8",
    )
    return job


def _write_reproduce(job: PreparedCorrJob, execution_mode: str) -> None:
    document: dict[str, Any] = yaml.safe_load(job.config.config_path.read_text(encoding="utf-8"))
    document["reproduction"] = {
        "execution_mode": execution_mode,
        "command": f"nematics corr {job.config.config_path} --{execution_mode}",
        "config_path": str(job.config.config_path),
        "fortran_source": str(_source_path(job.config)),
    }
    job.partial_reproduce_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def run_corr_local(job: PreparedCorrJob) -> tuple[Path, ...]:
    existing = [path for path in _all_final_paths(job) if path.exists()]
    if existing and not job.config.overwrite:
        raise FileExistsError("Refusing to replace existing output: " + ", ".join(map(str, existing)))
    for path in _all_partial_paths(job):
        path.unlink(missing_ok=True)
    _write_reproduce(job, "local")
    compiler = shutil.which(job.config.compiler)
    if compiler is None:
        raise RuntimeError(f"Fortran compiler was not found: {job.config.compiler}")
    with tempfile.TemporaryDirectory(prefix="nematics-corr-") as tmp:
        executable = Path(tmp) / "corr"
        subprocess.run(
            [compiler, *job.config.compiler_flags, str(_source_path(job.config)), "-o", str(executable)],
            check=True,
            cwd=job.config.repo_root,
        )
        subprocess.run([str(executable), str(job.namelist_path)], check=True, cwd=job.config.repo_root)
    if not all(path.is_file() for path in _all_partial_paths(job)):
        raise RuntimeError("Corr run completed without producing every output")
    for partial, final in zip(_all_partial_paths(job), _all_final_paths(job)):
        partial.replace(final)
    return _all_final_paths(job)


def submit_corr_job(job: PreparedCorrJob) -> str:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch was not found on PATH")
    _write_reproduce(job, "slurm")
    result = subprocess.run(
        [sbatch, str(job.slurm_script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
