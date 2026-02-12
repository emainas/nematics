# Troubleshooting

## `zsh: command not found: nematics`

Activate env and install editable package:

```bash
source .venv-nematics/bin/activate
uv pip install -e .
```

## `FileNotFoundError frame0.gro`

Use absolute path or run from repo root. The CLI now attempts repo-root fallback for relative paths.

## Paircorr is very slow

Use Fortran backend:

```bash
nematics build-fortran
nematics paircorr ... --backend fortran --save-every 0
```

## `Fortran backend requested but extension is not available`

Build extension first:

```bash
nematics build-fortran
```

If compiler missing, install `gfortran` and retry.

## Matplotlib/font cache warnings

Set writable cache paths:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/.cache
```

## `plot_ps.py` does not include your files

Ensure order filenames match expected stems:

- `gro_S.npy`
- `gro_2_S.npy`
- `gro_3_S.npy`
- `gro_4_S.npy`

`*_ddir_S.npy` files are intentionally separate and not used by default temperature map.

## MkDocs command not found

Install docs extras:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```
