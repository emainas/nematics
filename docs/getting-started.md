# Quickstart

## 1. Environment setup

From repo root:

```bash
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e ".[viz]"
```

If you also want documentation tooling:

```bash
uv pip install -e ".[docs]"
```

## 2. Set your paths

Define these before running commands:

- `<DATA_ROOT>`: directory containing trajectory folders
- `<GRO_DIR>`: one trajectory directory containing `frame*.gro`
- `<ORDER_DIR>`: where `*_S.npy` files will be written
- `<PAIRCORR_DIR>`: where paircorr `.dat` files will be written

Expected `<DATA_ROOT>` layout:

```text
<DATA_ROOT>/
  gro/
  gro 2/
  gro 3/
  gro 4/
```

Check quickly:

```bash
ls "<GRO_DIR>/frame0.gro"
```

## 3. First command: single-frame PAF

```bash
nematics paf \
  --gro-dir "<GRO_DIR>" \
  --frame 0
```

## 4. Compute order arrays (for report plots)

```bash
nematics order --data-root "<DATA_ROOT>" --axis molzax --out-dir "<ORDER_DIR>"
```

This produces:

- `<ORDER_DIR>/gro_S.npy`
- `<ORDER_DIR>/gro_2_S.npy`
- `<ORDER_DIR>/gro_3_S.npy`
- `<ORDER_DIR>/gro_4_S.npy`

## 5. Build fast paircorr backend

```bash
nematics build-fortran
```

## 6. Run pair correlations

```bash
for d in "gro" "gro 2" "gro 3" "gro 4"; do
  nematics paircorr \
    --gro-dir "<DATA_ROOT>/$d" \
    --label "${d// /_}" \
    --out-dir "<PAIRCORR_DIR>" \
    --backend fortran \
    --save-every 0
done
```

## 7. Generate figures

Order figures:

```bash
python plotting/plot_ps.py --order-dir "<ORDER_DIR>"
```

Paircorr figures:

```bash
python plotting/plot_paircorr.py --paircorr-dir "<PAIRCORR_DIR>" --out-dir "<REPORT_DIR>"
```

## Optional concrete example

In this repository:

- `<DATA_ROOT>` = `data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign`
- `<GRO_DIR>` = `data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro`
- `<ORDER_DIR>` = `data/order`
- `<PAIRCORR_DIR>` = `data/paircorr`
- `<REPORT_DIR>` = `reports/paircorr`
