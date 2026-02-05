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
nematics-paf --gro-dir "data/Poon, Chi-Duen - 5cb-x1000-gaff-88x88x94-zalign/gro" --frame 0
```

Plot P(S) across the four trajectories:

```bash
uv pip install -e ".[viz]"
python plotting/plot_ps.py
```

If you want to disable progress output:

```bash
python plotting/plot_ps.py --no-progress
```
