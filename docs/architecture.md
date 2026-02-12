# Architecture

## Package layout

- `src/nematics/gro.py`: GRO parsing and frame container
- `src/nematics/paf.py`: molecular axes + global PAF tensor eigensystem
- `src/nematics/order.py`: trajectory-level `S` computation and `.npy` outputs
- `src/nematics/histograms.py`: Fortran-style histogram data structures
- `src/nematics/paircorr.py`: pair-correlation orchestration and updates
- `src/nematics/_paircorr_fort.f90`: compiled inner loop for paircorr
- `src/nematics/fortran_backend.py`: f2py build/load helper
- `src/nematics/cli.py`: user-facing CLI

## Data flow

1. `read_gro_frame(...)` reads `frameN.gro` into positions + box.
2. `make_molecular_axes(...)` derives molecular frame and dipole vectors.
3. `compute_global_paf(...)` computes director from selected orientation vectors.
4. For `order`: largest eigenvalue per frame is written to `.npy`.
5. For `paircorr`: pair loops update 1D histograms with weighted channels.

## Paircorr output schema

Each `*.dat` file has shape `(200, 12)`:

- column 0: bin center
- column 1: normalized counts `g0`-like channel
- columns 2..11: weighted orientational channels from Fortran `WW`

In current plotting scripts, **column 2** (`WW(2)` channel) is used as `g1`.

## Fortran acceleration design

Python keeps I/O and orchestration. The expensive pair+image loops are offloaded:

- Python backend: `paircorr._paircorr_update`
- Fortran backend: `paircorr_fort_mod.paircorr_update`

This keeps behavior identical while reducing runtime by large factors.
