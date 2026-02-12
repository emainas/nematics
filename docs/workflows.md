# Workflows

## Workflow A: reproduce existing order report

Use this if your goal is to reproduce the current `P(S)` and average-`S` figures.

```bash
source .venv-nematics/bin/activate
nematics order --axis molzax --out-dir data/order
python plotting/plot_ps.py --order-dir data/order
```

Outputs:

- `reports/p_s_hist.png`
- `reports/avg_s_vs_t.png`

## Workflow B: CN-director variant

The 11-feb Fortran path often uses dipole-derived director context.

```bash
nematics order --axis ddir --out-dir data/order
```

This writes `*_ddir_S.npy` files.

## Workflow C: full pair-correlation pipeline

```bash
nematics build-fortran
for d in "gro" "gro 2" "gro 3" "gro 4"; do
  nematics paircorr \
    --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/$d" \
    --label "${d// /_}" \
    --out-dir data/paircorr \
    --backend fortran \
    --save-every 0
done
python plotting/plot_paircorr.py --paircorr-dir data/paircorr --out-dir reports/paircorr
```

## Workflow D: debug/smoke run (fast)

Use for validation before long jobs:

```bash
nematics paircorr \
  --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro" \
  --label smoke \
  --out-dir /tmp/paircorr_smoke \
  --nframes 1 \
  --save-every 0 \
  --backend auto
```

## Recommended execution policy

- Use `--backend fortran` for production runs.
- Keep `--save-every 0` unless you need crash recovery checkpoints.
- For very long jobs, run one directory at a time and archive `data/paircorr` outputs immediately.
