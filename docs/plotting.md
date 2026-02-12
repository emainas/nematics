# Plotting Guide

## 1. Order plots (`plot_ps.py`)

Command:

```bash
python plotting/plot_ps.py --order-dir "<ORDER_DIR>"
```

Input expected:

- `gro_S.npy`
- `gro_2_S.npy`
- `gro_3_S.npy`
- `gro_4_S.npy`

Temperature mapping in script:

- `gro_S` -> 356 K
- `gro_2_S` -> 366 K
- `gro_3_S` -> 376 K
- `gro_4_S` -> 386 K

Generated figures:

- `reports/p_s_hist.png`: distribution overlays of `S`
- `reports/avg_s_vs_t.png`: mean +/- std vs temperature

## 2. Paircorr plots (`plot_paircorr.py`)

Command:

```bash
python plotting/plot_paircorr.py --paircorr-dir "<PAIRCORR_DIR>" --out-dir "<REPORT_DIR>"
```

### Panels produced

- `paircorr_5cb_along_director.png`
- `paircorr_cn_along_director.png`
- `paircorr_5cb_perp_radial.png`
- `paircorr_cn_perp_radial.png`
- `paircorr_cn_dipole_radial.png`

### What is plotted now

`plot_paircorr.py` reads column 2 (`W_2`) from each `.dat` file as the plotted `g1` channel.

That is:

\[
g_1 \sim \langle \mathbf{u}_i\cdot\mathbf{u}_j \rangle
\]

for the species-specific orientation vectors.

## 3. If you want alternate channels

Edit `plotting/plot_paircorr.py` and change selected column index:

- col 2: `W_2` (dot product)
- col 3: `W_2^2`
- col 4: director-projected dot term
- col 5: square of director-projected term
- col 6: `y`-axis correlation

This can expose additional signatures (biaxial/tilt trends).

## 4. Isotropic S histogram with CLT/LDT overlays

Use `plot_s_isotropic_theory.py` to overlay:

- histogram of largest eigenvalue `S`
- central-limit-theory (CLT) curve
- large-deviation-theory (LDT) curve
- vertical line at mean histogram `S`

Recommended fast path (reuse precomputed arrays):

```bash
python plotting/plot_s_isotropic_theory.py \
  --s-file data/order/gro_4_S.npy \
  --chi-file data/susceptibility/gro_4_chi.npy \
  --num-particles 1000 \
  --out reports/s_isotropic_theory.png \
  --no-progress
```

Trajectory-driven path (recomputes `S` and `chi` first):

```bash
python plotting/plot_s_isotropic_theory.py \
  --gro-dir "<GRO_DIR_ISOTROPIC>" \
  --nframes 1000 \
  --axis molzax \
  --out reports/s_isotropic_theory.png
```

The CLT and LDT expressions and fitting equation are documented in
`docs/math/fluctuation-theory.md`.

Reference used for this theory overlay:

Mainas, E., & Stratt, R. M. *Exceptionally large fluctuations in orientational order: The lessons of large-deviation theory for liquid crystalline systems.* Journal of Chemical Physics, 162, 024501 (2025).
