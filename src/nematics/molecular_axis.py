from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_SAFE_SLURM_VALUE = re.compile(r"^[A-Za-z0-9_.:+@/-]+$")


@dataclass(frozen=True)
class MolecularAxisConfig:
    config_path: Path
    repo_root: Path
    molecule: str
    dataset: str
    run_id: str
    temperature_k: float
    trajectory: Path
    trajectory_format: str
    molecules: int
    atoms_per_molecule: int
    frame_start: int
    frame_stop: int
    frame_stride: int
    axis_name: str
    start_atom: int
    end_atom: int
    use_minimum_image: bool
    normalize: bool
    output_path: Path
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
class PreparedMolecularAxisJob:
    config: MolecularAxisConfig
    provenance_dir: Path
    namelist_path: Path
    slurm_script_path: Path
    normalized_config_path: Path
    metadata_path: Path
    reproduce_path: Path
    performance_path: Path
    partial_output_path: Path
    partial_metadata_path: Path
    partial_reproduce_path: Path
    partial_performance_path: Path


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


def _safe_slurm_value(value: Any, field: str) -> str:
    result = str(value)
    if not result or not _SAFE_SLURM_VALUE.fullmatch(result):
        raise ValueError(f"{field} contains unsupported characters: {result!r}")
    return result


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


def load_molecular_axis_config(path: str | Path) -> MolecularAxisConfig:
    """Read and validate one molecular-axis extraction YAML file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "root")

    version = int(root.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported molecular-axis YAML version: {version}")

    system = _mapping(_required(root, "system", "root"), "system")
    trajectory_cfg = _mapping(
        _required(root, "trajectory", "root"), "trajectory"
    )
    frames = _mapping(_required(root, "frames", "root"), "frames")
    axis = _mapping(_required(root, "molecular_axis", "root"), "molecular_axis")
    output = _mapping(_required(root, "output", "root"), "output")
    slurm = _mapping(_required(root, "slurm", "root"), "slurm")
    fortran = _mapping(root.get("fortran", {}), "fortran")

    trajectory = _resolve_path(
        _required(trajectory_cfg, "path", "trajectory"), config_path.parent
    )
    if not trajectory.is_file():
        raise FileNotFoundError(f"Trajectory does not exist: {trajectory}")

    trajectory_format = str(trajectory_cfg.get("format", "tinker_arc")).lower()
    if trajectory_format != "tinker_arc":
        raise ValueError("Only trajectory.format: tinker_arc is currently supported")

    molecules = _positive_int(
        _required(trajectory_cfg, "molecules", "trajectory"),
        "trajectory.molecules",
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

    start_atom = _positive_int(
        _required(axis, "start_atom", "molecular_axis"),
        "molecular_axis.start_atom",
    )
    end_atom = _positive_int(
        _required(axis, "end_atom", "molecular_axis"),
        "molecular_axis.end_atom",
    )
    if start_atom > atoms_per_molecule or end_atom > atoms_per_molecule:
        raise ValueError(
            "Molecular-axis atom indices must not exceed atoms_per_molecule"
        )
    if start_atom == end_atom:
        raise ValueError("Molecular-axis start_atom and end_atom must differ")

    pbc = str(axis.get("pbc", "minimum_image")).lower()
    if pbc not in {"minimum_image", "none"}:
        raise ValueError("molecular_axis.pbc must be 'minimum_image' or 'none'")

    output_path = _resolve_path(_required(output, "path", "output"), _repo_root())
    processed_root = (_repo_root() / "data-processed").resolve()
    _ensure_under(output_path, processed_root, "output.path")
    if output_path.suffix != ".bin":
        raise ValueError("output.path must end in .bin")

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
        raise ValueError(
            "The molecular-axis extractor is serial; slurm.cpus_per_task must be 1"
        )

    account_value = slurm.get("account")
    qos_value = slurm.get("qos")

    return MolecularAxisConfig(
        config_path=config_path,
        repo_root=_repo_root(),
        molecule=str(_required(system, "molecule", "system")),
        dataset=str(_required(system, "dataset", "system")),
        run_id=str(_required(system, "run_id", "system")),
        temperature_k=float(_required(system, "temperature_k", "system")),
        trajectory=trajectory,
        trajectory_format=trajectory_format,
        molecules=molecules,
        atoms_per_molecule=atoms_per_molecule,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_stride=frame_stride,
        axis_name=str(axis.get("name", "molecular-axis")),
        start_atom=start_atom,
        end_atom=end_atom,
        use_minimum_image=pbc == "minimum_image",
        normalize=_boolean(axis.get("normalize", True), "molecular_axis.normalize"),
        output_path=output_path,
        overwrite=_boolean(output.get("overwrite", False), "output.overwrite"),
        slurm_partition=_safe_slurm_value(
            slurm.get("partition", "general"), "slurm.partition"
        ),
        slurm_account=(
            _safe_slurm_value(account_value, "slurm.account")
            if account_value
            else None
        ),
        slurm_qos=(
            _safe_slurm_value(qos_value, "slurm.qos") if qos_value else None
        ),
        slurm_time=_safe_slurm_value(slurm.get("time", "02:00:00"), "slurm.time"),
        slurm_memory=_safe_slurm_value(slurm.get("memory", "2G"), "slurm.memory"),
        slurm_cpus_per_task=cpus,
        slurm_job_name=_safe_slurm_value(
            slurm.get("job_name", f"axis-{system.get('molecule', 'system')}"),
            "slurm.job_name",
        ),
        slurm_modules=tuple(modules_raw),
        compiler=str(fortran.get("compiler", "gfortran")),
        compiler_flags=tuple(flags_raw),
    )


def _fortran_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _bool(value: bool) -> str:
    return ".true." if value else ".false."


def _source_path(config: MolecularAxisConfig) -> Path:
    source = config.repo_root / "src" / "nematics" / "_molecular_axis.f90"
    if not source.is_file():
        raise FileNotFoundError(f"Fortran molecular-axis source is missing: {source}")
    return source


def _namelist_text(
    config: MolecularAxisConfig,
    partial_output: Path,
    partial_metadata: Path,
    partial_performance: Path,
) -> str:
    return "\n".join(
        [
            "&molecular_axis_job",
            f"  trajectory_path = {_fortran_string(config.trajectory)}",
            f"  output_path = {_fortran_string(partial_output)}",
            f"  final_output_path = {_fortran_string(config.output_path)}",
            f"  metadata_path = {_fortran_string(partial_metadata)}",
            f"  performance_path = {_fortran_string(partial_performance)}",
            f"  molecule_name = {_fortran_string(config.molecule)}",
            f"  dataset_name = {_fortran_string(config.dataset)}",
            f"  run_id = {_fortran_string(config.run_id)}",
            f"  axis_name = {_fortran_string(config.axis_name)}",
            f"  temperature_k = {config.temperature_k:.12g}",
            f"  molecules = {config.molecules}",
            f"  atoms_per_molecule = {config.atoms_per_molecule}",
            f"  frame_start = {config.frame_start}",
            f"  frame_stop = {config.frame_stop}",
            f"  frame_stride = {config.frame_stride}",
            f"  start_atom = {config.start_atom}",
            f"  end_atom = {config.end_atom}",
            f"  use_minimum_image = {_bool(config.use_minimum_image)}",
            f"  normalize_vectors = {_bool(config.normalize)}",
            "/",
            "",
        ]
    )


def _slurm_text(
    config: MolecularAxisConfig,
    namelist_path: Path,
    partial_output: Path,
    partial_metadata: Path,
    partial_reproduce: Path,
    partial_performance: Path,
    metadata_path: Path,
    reproduce_path: Path,
    performance_path: Path,
) -> str:
    source = _source_path(config)
    log_path = namelist_path.parent / "slurm-%j.out"
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

    compiler = shlex.quote(config.compiler)
    flags = " ".join(shlex.quote(flag) for flag in config.compiler_flags)
    output = shlex.quote(str(config.output_path))
    metadata = shlex.quote(str(metadata_path))
    reproduce = shlex.quote(str(reproduce_path))
    performance = shlex.quote(str(performance_path))
    partial = shlex.quote(str(partial_output))
    partial_meta = shlex.quote(str(partial_metadata))
    partial_reproduce_quoted = shlex.quote(str(partial_reproduce))
    partial_performance_quoted = shlex.quote(str(partial_performance))
    overwrite_test = "" if config.overwrite else (
        f"if [[ -e {output} || -e {metadata} || -e {reproduce} || "
        f"-e {performance} ]]; then\n"
        "  echo 'Refusing to replace an existing completed output.' >&2\n"
        "  exit 3\n"
        "fi\n"
    )
    move_flag = "-f" if config.overwrite else ""
    lines.extend(
        [
            overwrite_test.rstrip(),
            "work_root=${SLURM_TMPDIR:-/tmp}",
            'exe="$work_root/nematics-molecular-axis-${SLURM_JOB_ID}"',
            f"{compiler} {flags} {shlex.quote(str(source))} -o \"$exe\"",
            f"rm -f -- {partial} {partial_meta} {partial_performance_quoted}",
            f"srun \"$exe\" {shlex.quote(str(namelist_path))}",
            f"mv {move_flag} -- {partial} {output}".replace("mv  --", "mv --"),
            f"mv {move_flag} -- {partial_meta} {metadata}".replace(
                "mv  --", "mv --"
            ),
            f"mv {move_flag} -- {partial_performance_quoted} {performance}".replace(
                "mv  --", "mv --"
            ),
            f"mv {move_flag} -- {partial_reproduce_quoted} {reproduce}".replace(
                "mv  --", "mv --"
            ),
            "echo 'Molecular-axis extraction completed.'",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def prepare_molecular_axis_job(
    config_or_path: MolecularAxisConfig | str | Path,
) -> PreparedMolecularAxisJob:
    """Create the namelist and Slurm script without submitting the job."""
    config = (
        config_or_path
        if isinstance(config_or_path, MolecularAxisConfig)
        else load_molecular_axis_config(config_or_path)
    )
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output already exists and output.overwrite is false: {config.output_path}"
        )

    output_dir = config.output_path.parent
    provenance = output_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_path.with_suffix(".meta.yaml")
    reproduce_path = output_dir / "reproduce.yaml"
    performance_path = output_dir / "performance.csv"
    partial_output = config.output_path.with_suffix(config.output_path.suffix + ".partial")
    partial_metadata = metadata_path.with_suffix(metadata_path.suffix + ".partial")
    partial_reproduce = output_dir / "reproduce.yaml.partial"
    partial_performance = output_dir / "performance.csv.partial"
    namelist = provenance / "molecular-axis.nml"
    slurm_script = provenance / "molecular-axis.sbatch"
    normalized = provenance / "config.yaml"

    namelist.write_text(
        _namelist_text(
            config,
            partial_output,
            partial_metadata,
            partial_performance,
        ),
        encoding="utf-8",
    )
    slurm_script.write_text(
        _slurm_text(
            config,
            namelist,
            partial_output,
            partial_metadata,
            partial_reproduce,
            partial_performance,
            metadata_path,
            reproduce_path,
            performance_path,
        ),
        encoding="utf-8",
    )
    normalized.write_text(
        yaml.safe_dump(
            yaml.safe_load(config.config_path.read_text(encoding="utf-8")),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return PreparedMolecularAxisJob(
        config=config,
        provenance_dir=provenance,
        namelist_path=namelist,
        slurm_script_path=slurm_script,
        normalized_config_path=normalized,
        metadata_path=metadata_path,
        reproduce_path=reproduce_path,
        performance_path=performance_path,
        partial_output_path=partial_output,
        partial_metadata_path=partial_metadata,
        partial_reproduce_path=partial_reproduce,
        partial_performance_path=partial_performance,
    )


def _write_reproduce(job: PreparedMolecularAxisJob, execution_mode: str) -> None:
    config_document = yaml.safe_load(
        job.config.config_path.read_text(encoding="utf-8")
    )
    config_document["reproduction"] = {
        "execution_mode": execution_mode,
        "command": (
            f"nematics molecular-axis {job.config.config_path} "
            f"--{execution_mode}"
        ),
        "config_path": str(job.config.config_path),
        "trajectory_path": str(job.config.trajectory),
        "output_path": str(job.config.output_path),
        "fortran_source": str(_source_path(job.config)),
    }
    job.partial_reproduce_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )


def run_molecular_axis_local(
    job: PreparedMolecularAxisJob,
) -> tuple[Path, Path]:
    """Compile and run the Fortran extractor directly on the current host."""
    config = job.config
    if not config.overwrite:
        existing = [
            path
            for path in (
                config.output_path,
                job.metadata_path,
                job.reproduce_path,
                job.performance_path,
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to replace existing completed output: "
                + ", ".join(str(path) for path in existing)
            )

    job.partial_output_path.unlink(missing_ok=True)
    job.partial_metadata_path.unlink(missing_ok=True)
    job.partial_performance_path.unlink(missing_ok=True)
    _write_reproduce(job, "local")
    compiler = shutil.which(config.compiler)
    if compiler is None:
        raise RuntimeError(f"Fortran compiler was not found on PATH: {config.compiler}")

    with tempfile.TemporaryDirectory(prefix="nematics-molecular-axis-") as tmp:
        executable = Path(tmp) / "molecular-axis"
        compile_command = [
            compiler,
            *config.compiler_flags,
            str(_source_path(config)),
            "-o",
            str(executable),
        ]
        subprocess.run(compile_command, check=True, cwd=config.repo_root)
        subprocess.run(
            [str(executable), str(job.namelist_path)],
            check=True,
            cwd=config.repo_root,
        )

    required_partial_outputs = (
        job.partial_output_path,
        job.partial_metadata_path,
        job.partial_reproduce_path,
        job.partial_performance_path,
    )
    if not all(path.is_file() for path in required_partial_outputs):
        raise RuntimeError("Run completed without producing every partial output")
    job.partial_output_path.replace(config.output_path)
    job.partial_metadata_path.replace(job.metadata_path)
    job.partial_performance_path.replace(job.performance_path)
    job.partial_reproduce_path.replace(job.reproduce_path)
    return config.output_path, job.metadata_path


def submit_molecular_axis_job(job: PreparedMolecularAxisJob) -> str:
    """Submit a prepared molecular-axis job and return Slurm's response."""
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
