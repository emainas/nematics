#!/usr/bin/env python3
"""Compile and run Alekos's supplied 11-Feb-2026 5CB Fortran program.

The scientific source files are not edited.  A temporary copy of the main
program receives only three portability substitutions: the trajectory path and
temperature are read from command-line arguments, the temperature loop starts
at its sole supplied entry, and the requested number of frames replaces the
hard-coded 1000-frame limit.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


SOURCE_NAMES = (
    "class_histograms.f90",
    "CommonGlobalVariables.f90",
    "Eigen_3x3.f90",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_main(source: str) -> str:
    source = source.replace("STOP(' CHECK CloseIm ')", "STOP ' CHECK CloseIm '", 1)
    source = source.replace(
        "integer i, snap, groAtoms, molid, atid, atid2, tt, Ntemps",
        "integer i, snap, groAtoms, molid, atid, atid2, tt, Ntemps, NframesToRun",
        1,
    )
    hardcoded = re.compile(
        r"i=0\s*\n"
        r"i=i\+1; TempCh\(i\)='390'.*?\n"
        r"Ntemps = I",
        flags=re.DOTALL,
    )
    replacement = """i=1
call get_command_argument(1, TrajPa(i))
if (len_trim(TrajPa(i)) == 0) stop 'missing trajectory path'
if (TrajPa(i)(len_trim(TrajPa(i)):len_trim(TrajPa(i))) /= '/') &
    TrajPa(i) = trim(TrajPa(i))//'/'
call get_command_argument(2, ch2)
if (len_trim(ch2) == 0) stop 'missing temperature label'
TempCh(i)=ch2
call get_command_argument(3, fname)
read(fname,*) NframesToRun
Ntemps = 1"""
    source, count = hardcoded.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the hard-coded trajectory table")
    source = source.replace("do tt = 6, Ntemps, 1", "do tt = 1, Ntemps, 1", 1)
    source = source.replace(
        "do snap = 1, 1000  ! loop over the snapshots",
        "do snap = 1, NframesToRun  ! loop over the snapshots",
        1,
    )
    return source


def compile_program(source_dir: Path, build_dir: Path) -> Path:
    main_source = source_dir / "ChiDoon-5CBLefteris.f90"
    adapted_main = build_dir / "ChiDoon-5CBLefteris.portable.f90"
    adapted_main.write_text(adapt_main(main_source.read_text()))
    pair_source = (source_dir / "PairCorr_forHists.f90").read_text()
    for value in ("1.0", "4.0", "9.0", "25.0"):
        pair_source = pair_source.replace(f"DNZ*{value}D0", f"DNZ*{value}")
    adapted_pair = build_dir / "PairCorr_forHists.portable.f90"
    adapted_pair.write_text(pair_source)
    executable = build_dir / "alekos_5cb_exact"
    command = [
        "gfortran",
        "-O3",
        "-ffree-line-length-none",
        "-fallow-argument-mismatch",
        "-fdefault-real-8",
        "-fdefault-double-8",
        "-J",
        str(build_dir),
        "-o",
        str(executable),
        *(str(source_dir / name) for name in SOURCE_NAMES),
        str(adapted_pair),
        str(adapted_main),
    ]
    subprocess.run(command, cwd=build_dir, check=True)
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gro-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--nframes", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("legacy_fortran_code/11-feb-2026"),
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    gro_dir = args.gro_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not (gro_dir / "frame0.gro").is_file():
        raise SystemExit(f"No frame0.gro in {gro_dir}")
    if not 1 <= args.nframes <= 1000:
        raise SystemExit("--nframes must be between 1 and 1000")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alekos-5cb-exact-") as temporary:
        temporary_path = Path(temporary)
        executable = compile_program(source_dir, temporary_path)
        run_dir = temporary_path / "run"
        result_dir = run_dir / "5CB"
        result_dir.mkdir(parents=True)
        subprocess.run(
            [str(executable), str(gro_dir), str(args.temperature), str(args.nframes)],
            cwd=run_dir,
            check=True,
        )
        for result in result_dir.iterdir():
            shutil.copy2(result, out_dir / result.name)

    provenance = [
        "Alekos 5CB exact-source run",
        f"trajectory: {gro_dir}",
        f"temperature_K: {args.temperature}",
        f"frames: {args.nframes}",
        "portability changes: CLI trajectory/temperature/frame count and eight ",
        "  real-literal kind corrections required by modern gfortran",
        "compiler precision flags: -fdefault-real-8 -fdefault-double-8",
        "source hashes:",
    ]
    for name in (*SOURCE_NAMES, "PairCorr_forHists.f90", "ChiDoon-5CBLefteris.f90"):
        provenance.append(f"  {sha256(source_dir / name)}  {name}")
    (out_dir / "PROVENANCE.txt").write_text("\n".join(provenance) + "\n")
    print(f"Wrote Alekos outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
