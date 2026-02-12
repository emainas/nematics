# Results Report

This page summarizes generated analysis outputs as a compact report.

## Dataset and context

- System: 5CB trajectories (`gro`, `gro 2`, `gro 3`, `gro 4`)
- Frames analyzed: 1000 per trajectory
- Isotropic-focused theory analysis: `gro 4` (highest temperature)

## Key summary numbers (gro 4)

- Mean largest eigenvalue: **0.11123**
- Std largest eigenvalue: **0.02722**
- Mean susceptibility: **18.54695**
- Std susceptibility: **7.62422**
- Fitted theory parameter: **$\beta\epsilon = 0.806686$**

## Order parameter results

### Distribution of $S$

![P(S) distribution](assets/report/p_s_hist.png)

### Mean $S$ vs temperature

![Average S vs temperature](assets/report/avg_s_vs_t.png)

## Pair-correlation results

### 5CB pair correlations along director

![5CB along director](assets/report/paircorr_5cb_along_director.png)

### 5CB perpendicular and radial pair correlations

![5CB perpendicular and radial](assets/report/paircorr_5cb_perp_radial.png)

### CN pair correlations along director

![CN along director](assets/report/paircorr_cn_along_director.png)

### CN perpendicular and radial pair correlations

![CN perpendicular and radial](assets/report/paircorr_cn_perp_radial.png)

### CN dipole-dipole radial correlations

![CN dipole radial](assets/report/paircorr_cn_dipole_radial.png)

## Susceptibility result

### Time series of susceptibility $\chi(t)$ (gro 4)

![Susceptibility timeseries](assets/report/gro_4_chi_timeseries.png)

## Isotropic fluctuation-theory overlay

Histogram of largest eigenvalue $S$ with central-limit-theory (CLT) and large-deviation-theory (LDT) overlays:

![S histogram with CLT and LDT overlays](assets/report/s_isotropic_theory.png)

Theory details and equations are documented in:

- `Math -> CLT and LDT for S`

Reference:

Mainas, E., & Stratt, R. M. *Exceptionally large fluctuations in orientational order: The lessons of large-deviation theory for liquid crystalline systems.* Journal of Chemical Physics, 162, 024501 (2025).
