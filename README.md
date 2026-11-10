# nematics

Fortran-backed analysis of liquid-crystal molecular-dynamics trajectories.
Raw trajectories are external and read only; generated products belong under
`data-processed/<system>/<dataset>/<run>/<analysis>/`.

## Setup

```bash
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e ".[dev]"
```

`gfortran` and Slurm are required for compute-node execution.

## Commands

```bash
nematics molecular-axis CONFIG.yaml --slurm
nematics nematic-order CONFIG.yaml --slurm
nematics corr CONFIG.yaml --slurm
nematics dihedral CONFIG.yaml --local
```

Use `--local` instead of `--slurm` to run Fortran on the current host. The
cpptraj-backed dihedral command is currently local-only. With neither flag,
commands only prepare their reproducibility inputs.

Configurations mirror processed paths under `config/<system>/<dataset>/<run>/`.
Successful runs retain numerical output, metadata, `reproduce.yaml`,
`performance.csv`, and Slurm provenance.

## Layout

- `src/nematics/`: Python launchers and Fortran calculations
- `config/`: datasets and per-run YAML configurations
- `data-processed/`: generated numerical products
- `tests/`: focused validation tests
