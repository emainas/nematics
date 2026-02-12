# Workflows

## Workflow A: reproduce existing order report

Use this if your goal is to reproduce the current `P(S)` and average-`S` figures.

```bash
source .venv-nematics/bin/activate
nematics order --data-root "<DATA_ROOT>" --axis molzax --out-dir "<ORDER_DIR>"
python plotting/plot_ps.py --order-dir "<ORDER_DIR>"
```

Outputs:

- `reports/p_s_hist.png`
- `reports/avg_s_vs_t.png`

## Workflow B: CN-director variant

The 11-feb Fortran path often uses dipole-derived director context.

```bash
nematics order --data-root "<DATA_ROOT>" --axis ddir --out-dir "<ORDER_DIR>"
```

This writes `*_ddir_S.npy` files.

## Workflow C: full pair-correlation pipeline

```bash
nematics build-fortran
for d in "gro" "gro 2" "gro 3" "gro 4"; do
  nematics paircorr \
    --gro-dir "<DATA_ROOT>/$d" \
    --label "${d// /_}" \
    --out-dir "<PAIRCORR_DIR>" \
    --backend fortran \
    --save-every 0
done
python plotting/plot_paircorr.py --paircorr-dir "<PAIRCORR_DIR>" --out-dir "reports/paircorr"
```

## Workflow D: debug/smoke run (fast)

Use for validation before long jobs:

```bash
nematics paircorr \
  --gro-dir "<GRO_DIR>" \
  --label smoke \
  --out-dir /tmp/paircorr_smoke \
  --nframes 1 \
  --save-every 0 \
  --backend auto
```

## Recommended execution policy

- Use `--backend fortran` for production runs.
- Keep `--save-every 0` unless you need crash recovery checkpoints.
- For very long jobs, run one directory at a time and archive `<PAIRCORR_DIR>` outputs immediately.
