# nematics

Python translation of legacy Fortran analysis for 5CB Gromacs trajectories.

## Quick start

```bash
# Create env (requires uv)
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e .
```

Run on a single frame:

```bash
nematics paf --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro" --frame 0
```

Plot P(S) from saved order data:

```bash
uv pip install -e ".[viz]"
python plotting/plot_ps.py
```

Compute and save S for all four trajectories:

```bash
nematics order
```

## Example Outputs

![P(S) distribution](reports/p_s_hist.png)

![Average S vs Temperature](reports/avg_s_vs_t.png)

If you want to disable progress output:

```bash
python plotting/plot_ps.py --no-progress
```
