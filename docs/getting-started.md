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

## 2. Verify data path

The project expects trajectory folders under:

```text
data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/
  gro/
  gro 2/
  gro 3/
  gro 4/
```

Check quickly:

```bash
ls "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro/frame0.gro"
```

## 3. First command: single-frame PAF

```bash
nematics paf \
  --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro" \
  --frame 0
```

## 4. Compute order arrays (for report plots)

```bash
nematics order --axis molzax --out-dir data/order
```

This produces:

- `data/order/gro_S.npy`
- `data/order/gro_2_S.npy`
- `data/order/gro_3_S.npy`
- `data/order/gro_4_S.npy`

## 5. Build fast paircorr backend

```bash
nematics build-fortran
```

## 6. Run pair correlations

```bash
for d in "gro" "gro 2" "gro 3" "gro 4"; do
  nematics paircorr \
    --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/$d" \
    --label "${d// /_}" \
    --out-dir data/paircorr \
    --backend fortran \
    --save-every 0
done
```

## 7. Generate figures

Order figures:

```bash
python plotting/plot_ps.py --order-dir data/order
```

Paircorr figures:

```bash
python plotting/plot_paircorr.py --paircorr-dir data/paircorr --out-dir reports/paircorr
```
