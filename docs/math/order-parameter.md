# Math: Order Parameter and PAF

## Molecular orientation vectors

For each molecule, a unit orientation vector \(\mathbf{u}_i\) is defined from atom coordinates.

Two commonly used choices in this codebase:

- `molzax`: long molecular axis
- `ddir`: CN dipole direction

## Global PAF tensor

Given \(N\) molecules and orientation vectors \(\mathbf{u}_i=(u_{ix},u_{iy},u_{iz})\):

\[
\Omega_{\alpha\beta}=\sum_{i=1}^{N}u_{i\alpha}u_{i\beta}-\frac{N}{3}\delta_{\alpha\beta}
\]

This is the traceless alignment tensor used in legacy Fortran.

Code mapping: `nematics.paf.compute_global_paf`.

## Eigen decomposition

Solve:

\[
\Omega\,\mathbf{e}_k = \lambda_k\,\mathbf{e}_k
\]

Then scale eigenvalues:

\[
S_k = \frac{3}{2}\frac{\lambda_k}{N}
\]

The largest scaled eigenvalue \(S_1\) is the reported nematic order parameter \(S\).

## Practical interpretation

- \(S\approx 1\): strong orientational alignment
- \(S\approx 0\): weak orientational order / isotropic-like
- negative minor eigenvalues are expected due to tracelessness

## Plot connection

`plotting/plot_ps.py` uses saved `*_S.npy` arrays to produce:

- histogram \(P(S)\)
- average \(\langle S\rangle\) vs temperature

### \(P(S)\) from simulation

![P(S) distribution](../assets/report/p_s_hist.png)

### \(\langle S\rangle\) vs temperature

![Average S vs temperature](../assets/report/avg_s_vs_t.png)
