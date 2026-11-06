from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 and 3.10
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_REGISTRY = REPOSITORY_ROOT / "config" / "datasets.toml"
PACKAGED_REGISTRY = Path(__file__).with_name("datasets.toml")
DEFAULT_REGISTRY = (
    REPOSITORY_REGISTRY if REPOSITORY_REGISTRY.is_file() else PACKAGED_REGISTRY
)


@dataclass(frozen=True)
class ProjectPaths:
    raw_root: Path
    processed_root: Path
    reports_root: Path


@dataclass(frozen=True)
class Run:
    id: str
    temperature_k: float
    historical_processed_id: str | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    molecule: str
    method: str
    raw_subdir: str
    processed_subdir: str
    trajectory: str
    molecules: int
    atoms_per_molecule: int
    frames: int
    temperatures_inferred: bool
    runs: tuple[Run, ...]

    def raw_path(self, raw_root: Path | None = None) -> Path:
        root = raw_root or get_project_paths().raw_root
        return root / self.raw_subdir

    def processed_path(self, processed_root: Path | None = None) -> Path:
        root = processed_root or get_project_paths().processed_root
        return root / self.processed_subdir

    def run_trajectories(
        self, dataset_root: Path | None = None
    ) -> tuple[tuple[Run, Path], ...]:
        root = dataset_root or self.raw_path()
        trajectories = tuple(
            (run, root / run.id / self.trajectory) for run in self.runs
        )
        is_pattern = any(character in self.trajectory for character in "*?[")
        missing = [
            path
            for _, path in trajectories
            if not (
                any(candidate.is_file() for candidate in path.parent.glob(path.name))
                if is_pattern
                else path.is_file()
            )
        ]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing trajectories for {self.name}:\n{formatted}")
        return trajectories


def _resolve_repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    registry_path = Path(
        os.environ.get("NEMATICS_DATASETS_FILE", str(DEFAULT_REGISTRY))
    ).expanduser()
    with registry_path.open("rb") as handle:
        return tomllib.load(handle)


def get_project_paths() -> ProjectPaths:
    values = _load_registry()["paths"]
    raw_root = Path(os.environ.get("NEMATICS_RAW_ROOT", values["raw_root"])).expanduser()
    processed = os.environ.get("NEMATICS_PROCESSED_ROOT", values["processed_root"])
    reports = os.environ.get("NEMATICS_REPORTS_ROOT", values["reports_root"])
    return ProjectPaths(
        raw_root=raw_root,
        processed_root=_resolve_repository_path(processed),
        reports_root=_resolve_repository_path(reports),
    )


@lru_cache(maxsize=None)
def get_dataset(name: str) -> Dataset:
    try:
        values = _load_registry()["datasets"][name]
    except KeyError as exc:
        available = ", ".join(dataset_names())
        raise KeyError(f"Unknown dataset {name!r}; choose from: {available}") from exc

    runs = tuple(
        Run(
            id=run["id"],
            temperature_k=float(run["temperature_k"]),
            historical_processed_id=run.get("historical_processed_id"),
        )
        for run in values["runs"]
    )
    return Dataset(
        name=name,
        molecule=values["molecule"],
        method=values["method"],
        raw_subdir=values["raw_subdir"],
        processed_subdir=values["processed_subdir"],
        trajectory=values["trajectory"],
        molecules=int(values["molecules"]),
        atoms_per_molecule=int(values["atoms_per_molecule"]),
        frames=int(values["frames"]),
        temperatures_inferred=bool(values["temperatures_inferred"]),
        runs=runs,
    )


def dataset_names() -> tuple[str, ...]:
    return tuple(sorted(_load_registry()["datasets"]))
