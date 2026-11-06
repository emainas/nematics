# nematics

Research analysis code for liquid-crystal molecular-dynamics trajectories.
The current workflows cover 5CB monomers with GAFF and AMOEBA, plus CB7CB
conformation and induced-chirality analysis.

## Layout

- `src/nematics/`: reusable trajectory, order, susceptibility, and
  pair-correlation code
- `config/datasets.toml`: source-run and processed-layout registry
- `workflows/`: analysis programs grouped by molecular system
- `tools/`: trajectory-conversion and VMD utilities
- `tests/`: automated checks
- `data-processed/`: generated numerical products, ignored by Git
- `reports/`: generated figures and reports, ignored by Git

Raw trajectories remain external and read only. Generated files belong under
`data-processed/<molecule>/<dataset>/<workflow>/`.

## Setup

```bash
uv venv .venv-nematics
source .venv-nematics/bin/activate
uv pip install -e ".[dev,viz]"
```

Use command help for workflow options:

```bash
nematics --help
nematics datasets --check
python workflows/<system>/<workflow>.py --help
```

AMOEBA workflows generally require AmberTools `cpptraj`. The optional compiled
pair-correlation backend requires `gfortran` and can be built with
`nematics build-fortran`.
