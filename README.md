# nematics

Analysis tools for liquid-crystal molecular-dynamics trajectories. Raw data
are external and read only; generated products stay under `data-processed/`.

## Setup

```bash
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e ".[dev,viz]"
```

Fortran commands require `gfortran`; Slurm is needed only with `--slurm`.
Dihedral configurations contain the cpptraj executable path.

## Commands

```bash
nematics molecular-axis CONFIG.yaml --slurm
nematics nematic-order CONFIG.yaml --slurm
nematics corr CONFIG.yaml --slurm
nematics dihedral CONFIG.yaml --local
```

Use `--local` instead of `--slurm` to run Fortran on the current host. The
cpptraj-backed dihedral command is local-only. With neither execution flag,
commands prepare their reproducibility inputs without running the analysis.

Current summary figures are generated with:

```bash
python tools/plot_nematic_order_vs_temperature.py
python tools/plot_corr_panels.py
python tools/plot_4ocb_dihedral_pmfs.py
```

Configurations mirror processed paths under `config/<system>/<dataset>/<run>/`.
Successful runs retain numerical output, metadata, `reproduce.yaml`,
`performance.csv`, and execution provenance.

Correlation YAMLs may request one `analysis.rank` or `analysis.ranks: [0, 1, 2]`;
the latter writes `g0`, `g1`, `g2`, `C1`, and `C2` from one pair traversal.

## Layout

- `src/nematics/`: CLI, Python launchers, and Fortran calculations
- `config/`: datasets and per-run YAML configurations
- `data-processed/`: generated numerical products
- `reports/`: generated figures
- `tools/`: plotting utilities
- `tests/`: focused validation tests
