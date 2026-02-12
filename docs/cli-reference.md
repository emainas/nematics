# CLI Reference

The entrypoint is `nematics`.

## `nematics paf`

Compute PAF eigenvalues/eigenvectors for one frame.

```bash
nematics paf --gro-dir <path> [--frame 0] [--natoms-per-mol 38] [--axis molzax|ddir]
```

Arguments:

- `--gro-dir`: directory containing `frame*.gro`
- `--frame`: frame index (default `0`)
- `--natoms-per-mol`: atom count per molecule (default `38`)
- `--axis`: orientation set for PAF (`molzax` or `ddir`)

## `nematics order`

Compute largest eigenvalue `S` over all frames and default trajectory folders.

```bash
nematics order \
  [--data-root data/Poon,...] \
  [--out-dir data/order] \
  [--nframes 1000] \
  [--natoms-per-mol 38] \
  [--axis molzax|ddir] \
  [--no-progress]
```

## `nematics paircorr`

Compute Fortran-style pair-correlation histograms.

```bash
nematics paircorr \
  --gro-dir <path> \
  [--out-dir data/paircorr] \
  [--label <name>] \
  [--start-frame 0] \
  [--nframes 1000] \
  [--save-every 0] \
  [--natoms-per-mol 38] \
  [--backend auto|fortran|python] \
  [--auto-build-fortran] \
  [--no-progress]
```

Backend behavior:

- `fortran`: fail if extension is unavailable
- `python`: pure Python fallback
- `auto`: use Fortran if available, otherwise Python

## `nematics build-fortran`

Compile pair-correlation extension module.

```bash
nematics build-fortran [--quiet]
```

Output:

- shared object under `src/nematics/_paircorr_fort*.so`
