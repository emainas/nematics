#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FRAME_RE = re.compile(r"^frame(\d+)\.gro$")


@dataclass(frozen=True)
class Atom:
    name: str
    x: float
    y: float
    z: float


def frame_index(path: Path) -> int | None:
    match = FRAME_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def iter_frame_paths(gro_dir: Path, start: int | None, stop: int | None, stride: int) -> Iterable[Path]:
    paths: list[tuple[int, Path]] = []
    for path in gro_dir.glob("frame*.gro"):
        idx = frame_index(path)
        if idx is None:
            continue
        if start is not None and idx < start:
            continue
        if stop is not None and idx >= stop:
            continue
        if idx % stride != 0:
            continue
        paths.append((idx, path))

    for _, path in sorted(paths):
        yield path


def parse_atom_line(line: str) -> tuple[str, float, float, float]:
    try:
        atom_name = line[10:15].strip()
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        return atom_name, x, y, z
    except ValueError:
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed GRO atom line: {line!r}")
        return parts[1], float(parts[-3]), float(parts[-2]), float(parts[-1])


def xyz_symbol(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return "X"

    # VMD's XYZ reader is happiest with element-like labels. GRO atom names for
    # this project are mostly C/H/N-style names with numeric suffixes.
    upper = letters.upper()
    if upper.startswith("CL"):
        return "Cl"
    if upper.startswith("BR"):
        return "Br"
    return upper[0]


def read_gro_atoms(path: Path, scale: float) -> tuple[str, list[Atom]]:
    with path.open() as handle:
        title = handle.readline().rstrip("\n")
        natoms = int(handle.readline().strip())

        atoms: list[Atom] = []
        for _ in range(natoms):
            name, x, y, z = parse_atom_line(handle.readline())
            atoms.append(Atom(xyz_symbol(name), x * scale, y * scale, z * scale))

    return title, atoms


def write_xyz_trajectory(
    gro_dir: Path,
    out_path: Path,
    *,
    start: int | None,
    stop: int | None,
    stride: int,
    scale: float,
) -> int:
    frame_paths = list(iter_frame_paths(gro_dir, start=start, stop=stop, stride=stride))
    if not frame_paths:
        raise FileNotFoundError(f"No frame*.gro files found in {gro_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference_natoms: int | None = None
    with out_path.open("w") as out:
        for frame_no, frame_path in enumerate(frame_paths):
            title, atoms = read_gro_atoms(frame_path, scale=scale)
            if reference_natoms is None:
                reference_natoms = len(atoms)
            elif len(atoms) != reference_natoms:
                raise ValueError(
                    f"{frame_path} has {len(atoms)} atoms; expected {reference_natoms}"
                )

            out.write(f"{len(atoms)}\n")
            out.write(f"{frame_path.name} | {title} | source={gro_dir}\n")
            for atom in atoms:
                out.write(f"{atom.name:2s} {atom.x:12.6f} {atom.y:12.6f} {atom.z:12.6f}\n")

            if frame_no and frame_no % 100 == 0:
                print(f"Wrote {frame_no} frames...")

    return len(frame_paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a directory of frame*.gro files to a multi-frame XYZ trajectory for VMD."
    )
    parser.add_argument(
        "gro_dir",
        type=Path,
        help="Input directory containing frame0.gro, frame1.gro, ...",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("trajectory.xyz"),
        help="Output XYZ trajectory path (default: trajectory.xyz)",
    )
    parser.add_argument("--start", type=int, default=None, help="First frame index to include")
    parser.add_argument("--stop", type=int, default=None, help="Stop before this frame index")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame index")
    parser.add_argument(
        "--keep-nm",
        action="store_true",
        help="Write GRO coordinates in nm instead of converting to Angstrom",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    gro_dir = args.gro_dir.expanduser()
    scale = 1.0 if args.keep_nm else 10.0
    nframes = write_xyz_trajectory(
        gro_dir,
        args.out,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        scale=scale,
    )

    units = "nm" if args.keep_nm else "Angstrom"
    print(f"Wrote {nframes} frames to {args.out} ({units})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
