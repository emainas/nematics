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
