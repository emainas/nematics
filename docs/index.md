# nematics Documentation

This repository contains a Python translation of legacy Fortran workflows for nematic liquid crystal trajectory analysis, including:

- global orientational order parameter (`S`) from GRO trajectories
- Fortran-accelerated pair-correlation analysis (5CB and CN dipole channels)
- report plotting scripts for order and pair-correlation families

This documentation is written for two audiences:

1. first-time users who want an exact runbook
2. technical users who want implementation and math details

## Documentation map

- **Results Report**: ready-to-share figures and summary metrics
- **Quickstart**: setup + first successful run
- **Workflows**: reproducible command sequences for full analysis
- **CLI Reference**: all commands and arguments
- **Architecture**: module-level implementation map
- **Math**: precise definitions and what each output column means
- **Plotting Guide**: how report figures are built from generated data
- **Troubleshooting**: common failure modes and fixes

## One-minute sanity check

```bash
source .venv-nematics/bin/activate
uv pip install -e ".[viz]"
nematics paf --gro-dir "<GRO_DIR>" --frame 0
```

If this runs and prints three eigenvalues/eigenvectors, your environment is healthy.
