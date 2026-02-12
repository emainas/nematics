# Math: CLT and LDT for S

This page documents the two theory curves overlaid in `plotting/plot_s_isotropic_theory.py`.

## Inputs

The model uses:

- number of molecules: $N$
- mean susceptibility from trajectory: $\chi = \langle \chi_t \rangle$
- fit parameter: $\beta\epsilon$

with

$$
\chi_t = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\frac{1}{2}\left(3(\mathbf{u}_i\cdot\mathbf{u}_j)^2-1\right)
$$

where $\mathbf{u}_i$ are unit molecular orientation vectors (typically `molzax` in isotropic analysis).

## Central-limit approximation

The script uses

$$
a = \frac{5N}{3\chi}
$$

and the CLT curve

$$
P_{\mathrm{CLT}}(s)=\sqrt{\frac{6a}{\pi}}
\left[
\exp(-6as^2)
-
\left(1-\frac{9}{2}as^2\right)\exp\left(-\frac{3}{2}as^2\right)
\right].
$$

After evaluation, the code clips negative values to zero and normalizes the curve numerically to unit area.

## Large-deviation approximation

For each $s\in[0,0.5]$, with integration variable $x\in[-s/2,s]$:

$$
q(x,s)=\sqrt{\frac{2}{3}}\sqrt{2s^2+2x^2+2sx}
$$

$$
J(x,s)=(s-x)(2s+x)(s+2x)
$$

$$
\Phi(q)= -\ln(1-q)-q + \left(-\frac12+\frac{5}{2\chi}\right)q^2
-\left(\frac79-\frac{5}{3\chi}-\frac{\beta\epsilon}{3}\right)q^3
$$

$$
P_{\mathrm{raw}}(s)=\int_{-s/2}^{s} J(x,s)\,\exp\left[-N\,\Phi(q(x,s))\right]dx
$$

and finally

$$
P_{\mathrm{LDT}}(s)=\frac{P_{\mathrm{raw}}(s)}{\int P_{\mathrm{raw}}(s')\,ds'}.
$$

The script also reports the corresponding rate function:

$$
I(s)=-\frac{1}{N}\ln\left(P_{\mathrm{LDT}}(s)+\varepsilon\right),\quad \varepsilon=10^{-10}.
$$

## Fitting $\beta\epsilon$

$\beta\epsilon$ is solved numerically from

$$
\langle S\rangle_{\mathrm{hist}}-\langle S\rangle_{\mathrm{LDT}}(\beta\epsilon)=0.
$$

where

$$
\langle S\rangle_{\mathrm{LDT}}=\int_0^{0.5} s\,P_{\mathrm{LDT}}(s)\,ds.
$$

## Reference

Mainas, E., & Stratt, R. M. *Exceptionally large fluctuations in orientational order: The lessons of large-deviation theory for liquid crystalline systems.* Journal of Chemical Physics, 162, 024501 (2025).
