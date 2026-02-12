# nematics

Python translation of legacy Fortran analysis for 5CB Gromacs trajectories.

Documentation is available [here](docs/index.md).
Results report with generated figures is [here](docs/results-report.md).

## Quick start

```bash
# Create env (requires uv)
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e .
```

Run on a single frame:

```bash
nematics paf --gro-dir "<GRO_DIR>" --frame 0
```

Plot P(S) from saved order data:

```bash
uv pip install -e ".[viz]"
python plotting/plot_ps.py
```

Compute and save S for all four trajectories:

```bash
nematics order --data-root "<DATA_ROOT>" --out-dir "<ORDER_DIR>"
```

Recompute order with the `11-feb-2026` CN-dipole director (`ddir`) instead of the
legacy molecular long axis (`molzax`):

```bash
nematics order --data-root "<DATA_ROOT>" --axis ddir --out-dir "<ORDER_DIR>"
```

Compute the new `11-feb-2026` Fortran-style pair-correlation histograms for one trajectory:

```bash
# one-time build of fast Fortran backend
nematics build-fortran

nematics paircorr \
  --gro-dir "<GRO_DIR>" \
  --label "<RUN_LABEL>" \
  --out-dir "<PAIRCORR_DIR>" \
  --backend auto \
  --save-every 0
```

`--backend auto` uses Fortran if available (falls back to Python).  
Use `--backend fortran` to force compiled mode.

Compute orientational susceptibility time series (from molecular orientation vectors):

```bash
nematics susceptibility \
  --gro-dir "<GRO_DIR>" \
  --nframes 1000 \
  --axis molzax \
  --out "data/susceptibility/chi.npy"
```

Plot susceptibility time series and mean:

```bash
python plotting/plot_susceptibility.py \
  --chi-file "data/susceptibility/chi.npy" \
  --out "reports/susceptibility_timeseries.png"
```

For isotropic-phase analysis (typically `gro 4`), plot the histogram of the largest
eigenvalue `S` with CLT and LDT overlays:

```bash
python plotting/plot_s_isotropic_theory.py \
  --s-file data/order/gro_4_S.npy \
  --chi-file data/susceptibility/gro_4_chi.npy \
  --num-particles 1000 \
  --out "reports/s_isotropic_theory.png"
```

Theory reference for CLT/LDT overlay:

Mainas, E., & Stratt, R. M. *Exceptionally large fluctuations in orientational order: The lessons of large-deviation theory for liquid crystalline systems.* Journal of Chemical Physics, 162, 024501 (2025).

The generated `.dat` files include:
- `*_5CB_par_1.0.dat`, `*_5CB_par_2.0.dat`, `*_5CB_par_3.0.dat`, `*_5CB_par_all.dat`, `*_5CB_per.dat`, `*_5CB_rad.dat`
- `*_CN_par_1.0.dat`, `*_CN_par_2.0.dat`, `*_CN_par_3.0.dat`, `*_CN_par_all.dat`, `*_CN_per.dat`, `*_CN_rad.dat`

## Example Outputs

![P(S) distribution](reports/p_s_hist.png)

![Average S vs Temperature](reports/avg_s_vs_t.png)

If you want to disable progress output:

```bash
python plotting/plot_ps.py --no-progress
```

To reproduce the current report figures from saved order arrays:

```bash
python plotting/plot_ps.py --order-dir "<ORDER_DIR>"
```

To generate pair-correlation analysis figures from `<PAIRCORR_DIR>/*.dat`:

```bash
python plotting/plot_paircorr.py --paircorr-dir "<PAIRCORR_DIR>" --out-dir "<REPORT_DIR>"
```

Example values in this repo:
- `<DATA_ROOT>`: `data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign`
- `<GRO_DIR>`: `data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro`
- `<ORDER_DIR>`: `data/order`
- `<PAIRCORR_DIR>`: `data/paircorr`
- `<REPORT_DIR>`: `reports/paircorr`

## Documentation (MkDocs)

Install docs dependencies and serve locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

Then open the local URL printed by MkDocs (usually `http://127.0.0.1:8000`).
