from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ext_suffix() -> str:
    import sysconfig

    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def build_paircorr_fortran(verbose: bool = True) -> Path:
    """Build the Fortran paircorr extension into src/nematics."""
    repo = _repo_root()
    src = repo / "src" / "nematics" / "_paircorr_fort.f90"
    if not src.exists():
        raise FileNotFoundError(src)

    with tempfile.TemporaryDirectory(prefix="nematics_f2py_") as td:
        td_path = Path(td)
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            str(src),
            "-m",
            "_paircorr_fort",
        ]
        proc = subprocess.run(
            cmd,
            cwd=td,
            capture_output=not verbose,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            msg = "Failed to build _paircorr_fort with f2py"
            if not verbose:
                msg += f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            raise RuntimeError(msg)

        suffix = _ext_suffix()
        built = sorted(td_path.glob(f"_paircorr_fort*{suffix}"))
        if not built:
            built = sorted(td_path.glob("_paircorr_fort*.so"))
        if not built:
            raise RuntimeError("f2py build succeeded but extension file was not found")

        target = repo / "src" / "nematics" / built[0].name
        shutil.copy2(built[0], target)

    importlib.invalidate_caches()
    return target


def load_paircorr_fortran(auto_build: bool = False):
    try:
        return importlib.import_module("nematics._paircorr_fort")
    except Exception:
        if not auto_build:
            return None

    build_paircorr_fortran(verbose=True)
    importlib.invalidate_caches()
    return importlib.import_module("nematics._paircorr_fort")
