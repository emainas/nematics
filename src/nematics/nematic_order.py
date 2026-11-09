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
class NematicOrderConfig:
    config_path: Path
    repo_root: Path
    molecule: str
    dataset: str
    run_id: str
    temperature_k: float
    input_path: Path
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
class PreparedNematicOrderJob:
    config: NematicOrderConfig
    provenance_dir: Path
    namelist_path: Path
    slurm_script_path: Path
    normalized_config_path: Path
    csv_path: Path
    metadata_path: Path
    reproduce_path: Path
    performance_path: Path
    partial_output_path: Path
    partial_csv_path: Path
    partial_metadata_path: Path
    partial_reproduce_path: Path
    partial_performance_path: Path


def load_nematic_order_config(path: str | Path) -> NematicOrderConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle), "root")
    version = int(root.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported nematic-order YAML version: {version}")

    system = _mapping(_required(root, "system", "root"), "system")
    input_config = _mapping(_required(root, "input", "root"), "input")
    output_config = _mapping(_required(root, "output", "root"), "output")
    slurm = _mapping(_required(root, "slurm", "root"), "slurm")
    fortran = _mapping(root.get("fortran", {}), "fortran")

    repo = _repo_root()
    processed_root = (repo / "data-processed").resolve()
    input_path = _resolve_path(_required(input_config, "path", "input"), repo)
    output_path = _resolve_path(_required(output_config, "path", "output"), repo)
    _ensure_under(input_path, processed_root, "input.path")
    _ensure_under(output_path, processed_root, "output.path")
    if not input_path.is_file():
        raise FileNotFoundError(f"Molecular-axis matrix does not exist: {input_path}")
    if input_path.suffix != ".bin" or output_path.suffix != ".bin":
        raise ValueError("input.path and output.path must end in .bin")
    if input_path == output_path:
        raise ValueError("input.path and output.path must differ")

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
        raise ValueError("The nematic-order extractor is serial; cpus_per_task must be 1")

    account = slurm.get("account")
    qos = slurm.get("qos")
    return NematicOrderConfig(
        config_path=config_path,
        repo_root=repo,
        molecule=str(_required(system, "molecule", "system")),
        dataset=str(_required(system, "dataset", "system")),
        run_id=str(_required(system, "run_id", "system")),
        temperature_k=float(_required(system, "temperature_k", "system")),
        input_path=input_path,
        output_path=output_path,
        overwrite=_boolean(output_config.get("overwrite", False), "output.overwrite"),
        slurm_partition=_safe_slurm_value(
            slurm.get("partition", "general"), "slurm.partition"
        ),
        slurm_account=(
            _safe_slurm_value(account, "slurm.account") if account else None
        ),
        slurm_qos=_safe_slurm_value(qos, "slurm.qos") if qos else None,
        slurm_time=_safe_slurm_value(slurm.get("time", "00:10:00"), "slurm.time"),
        slurm_memory=_safe_slurm_value(slurm.get("memory", "512M"), "slurm.memory"),
        slurm_cpus_per_task=cpus,
        slurm_job_name=_safe_slurm_value(
            slurm.get("job_name", f"order-{system.get('molecule', 'system')}"),
            "slurm.job_name",
        ),
        slurm_modules=tuple(modules_raw),
        compiler=str(fortran.get("compiler", "gfortran")),
        compiler_flags=tuple(flags_raw),
    )


def _source_path(config: NematicOrderConfig) -> Path:
    source = config.repo_root / "src" / "nematics" / "_nematic_order.f90"
    if not source.is_file():
        raise FileNotFoundError(f"Fortran nematic-order source is missing: {source}")
    return source


def _namelist_text(job: PreparedNematicOrderJob) -> str:
    config = job.config
    return "\n".join(
        [
            "&nematic_order_job",
            f"  input_path = {_fortran_string(config.input_path)}",
            f"  output_path = {_fortran_string(job.partial_output_path)}",
            f"  final_output_path = {_fortran_string(config.output_path)}",
            f"  csv_path = {_fortran_string(job.partial_csv_path)}",
            f"  metadata_path = {_fortran_string(job.partial_metadata_path)}",
            f"  performance_path = {_fortran_string(job.partial_performance_path)}",
            f"  molecule_name = {_fortran_string(config.molecule)}",
            f"  dataset_name = {_fortran_string(config.dataset)}",
            f"  run_id = {_fortran_string(config.run_id)}",
            f"  temperature_k = {config.temperature_k:.12g}",
            "/",
            "",
        ]
    )


def _slurm_text(job: PreparedNematicOrderJob) -> str:
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

    finals = (
        config.output_path,
        job.csv_path,
        job.metadata_path,
        job.performance_path,
        job.reproduce_path,
    )
    partials_runtime = (
        job.partial_output_path,
        job.partial_csv_path,
        job.partial_metadata_path,
        job.partial_performance_path,
    )
    if not config.overwrite:
        tests = " || ".join(f"-e {shlex.quote(str(path))}" for path in finals)
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
    lines.extend(
        [
            "work_root=${SLURM_TMPDIR:-/tmp}",
            'exe="$work_root/nematics-nematic-order-${SLURM_JOB_ID}"',
            f"{compiler} {flags} {shlex.quote(str(_source_path(config)))} -o \"$exe\"",
            "rm -f -- " + " ".join(shlex.quote(str(path)) for path in partials_runtime),
            f"srun \"$exe\" {shlex.quote(str(job.namelist_path))}",
        ]
    )
    move_flag = "-f" if config.overwrite else ""
    pairs = (
        (job.partial_output_path, config.output_path),
        (job.partial_csv_path, job.csv_path),
        (job.partial_metadata_path, job.metadata_path),
        (job.partial_performance_path, job.performance_path),
        (job.partial_reproduce_path, job.reproduce_path),
    )
    for partial, final in pairs:
        command = f"mv {move_flag} -- {shlex.quote(str(partial))} {shlex.quote(str(final))}"
        lines.append(command.replace("mv  --", "mv --"))
    lines.extend(["echo 'Nematic-order calculation completed.'", ""])
    return "\n".join(lines)


def prepare_nematic_order_job(
    config_or_path: NematicOrderConfig | str | Path,
) -> PreparedNematicOrderJob:
    config = (
        config_or_path
        if isinstance(config_or_path, NematicOrderConfig)
        else load_nematic_order_config(config_or_path)
    )
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output already exists and output.overwrite is false: {config.output_path}"
        )

    output_dir = config.output_path.parent
    provenance = output_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_path.with_suffix(".csv")
    metadata_path = config.output_path.with_suffix(".meta.yaml")
    reproduce_path = output_dir / "reproduce.yaml"
    performance_path = output_dir / "performance.csv"
    partial_output = config.output_path.with_suffix(".bin.partial")
    partial_csv = csv_path.with_suffix(".csv.partial")
    partial_metadata = metadata_path.with_suffix(".yaml.partial")
    partial_reproduce = output_dir / "reproduce.yaml.partial"
    partial_performance = output_dir / "performance.csv.partial"
    job = PreparedNematicOrderJob(
        config=config,
        provenance_dir=provenance,
        namelist_path=provenance / "nematic-order.nml",
        slurm_script_path=provenance / "nematic-order.sbatch",
        normalized_config_path=provenance / "config.yaml",
        csv_path=csv_path,
        metadata_path=metadata_path,
        reproduce_path=reproduce_path,
        performance_path=performance_path,
        partial_output_path=partial_output,
        partial_csv_path=partial_csv,
        partial_metadata_path=partial_metadata,
        partial_reproduce_path=partial_reproduce,
        partial_performance_path=partial_performance,
    )
    job.namelist_path.write_text(_namelist_text(job), encoding="utf-8")
    job.slurm_script_path.write_text(_slurm_text(job), encoding="utf-8")
    job.normalized_config_path.write_text(
        yaml.safe_dump(
            yaml.safe_load(config.config_path.read_text(encoding="utf-8")),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return job


def _write_reproduce(job: PreparedNematicOrderJob, execution_mode: str) -> None:
    document: dict[str, Any] = yaml.safe_load(
        job.config.config_path.read_text(encoding="utf-8")
    )
    document["reproduction"] = {
        "execution_mode": execution_mode,
        "command": (
            f"nematics nematic-order {job.config.config_path} --{execution_mode}"
        ),
        "config_path": str(job.config.config_path),
        "input_path": str(job.config.input_path),
        "output_path": str(job.config.output_path),
        "fortran_source": str(_source_path(job.config)),
    }
    job.partial_reproduce_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def run_nematic_order_local(job: PreparedNematicOrderJob) -> tuple[Path, Path]:
    config = job.config
    finals = (
        config.output_path,
        job.csv_path,
        job.metadata_path,
        job.performance_path,
        job.reproduce_path,
    )
    existing = [path for path in finals if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Refusing to replace existing output: "
            + ", ".join(str(path) for path in existing)
        )
    for path in (
        job.partial_output_path,
        job.partial_csv_path,
        job.partial_metadata_path,
        job.partial_performance_path,
    ):
        path.unlink(missing_ok=True)
    _write_reproduce(job, "local")
    compiler = shutil.which(config.compiler)
    if compiler is None:
        raise RuntimeError(f"Fortran compiler was not found on PATH: {config.compiler}")
    with tempfile.TemporaryDirectory(prefix="nematics-nematic-order-") as tmp:
        executable = Path(tmp) / "nematic-order"
        subprocess.run(
            [
                compiler,
                *config.compiler_flags,
                str(_source_path(config)),
                "-o",
                str(executable),
            ],
            check=True,
            cwd=config.repo_root,
        )
        subprocess.run(
            [str(executable), str(job.namelist_path)],
            check=True,
            cwd=config.repo_root,
        )
    partials = (
        job.partial_output_path,
        job.partial_csv_path,
        job.partial_metadata_path,
        job.partial_performance_path,
        job.partial_reproduce_path,
    )
    if not all(path.is_file() for path in partials):
        raise RuntimeError("Run completed without producing every partial output")
    for partial, final in zip(partials, finals):
        partial.replace(final)
    return config.output_path, job.csv_path


def submit_nematic_order_job(job: PreparedNematicOrderJob) -> str:
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
