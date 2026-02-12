from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

# Force non-interactive backend for scripts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nematics.gro import read_gro_frame
from nematics.order import compute_order_for_dir
from nematics.susceptibility import compute_susceptibility_timeseries


def simpson_like(y: np.ndarray, x: np.ndarray) -> float:
    # Robust, dependency-free integration helper.
    return float(np.trapezoid(y, x))


class AnalyticalMethods:
    def __init__(self, num_particles: int, mean_chi: float, beta_epsilon: float):
        self.N = num_particles
        self.chi = mean_chi
        self.be = beta_epsilon
        self.a = (5.0 * self.N) / (3.0 * self.chi)
        self.f = self.chi / self.N
        self.corr = 0.5
        self.C = (self.N / 4.0) * (2.0 / self.chi - 1.0 - self.corr)

    def central_limit_3D(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(6.0 * self.a / np.pi) * (
            np.exp(-6.0 * self.a * x * x)
            - (1.0 - 4.5 * self.a * x * x) * np.exp(-1.5 * self.a * x * x)
        )

    def large_deviations_3D(self, m: int = 500, nx: int = 350) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        small_value = 1e-10
        raw_prob = np.zeros(m, dtype=float)
        order_parameter = np.linspace(0.0, 0.5, m)

        for i in range(m):
            s = float(order_parameter[i])
            x_grid = np.linspace(-s / 2.0, s, nx)

            order = np.sqrt(2.0 / 3.0) * np.sqrt(2.0 * s * s + 2.0 * x_grid * x_grid + 2.0 * s * x_grid)
            order = np.clip(order, 0.0, 1.0 - 1e-12)

            exponent = -self.N * (
                -np.log1p(-order)
                - order
                + (-0.5 + 5.0 / (2.0 * self.chi)) * order * order
                - (7.0 / 9.0 - 5.0 / (3.0 * self.chi) - self.be / 3.0) * order * order * order
            )

            exponent = np.clip(exponent, -700.0, 700.0)
            pade = np.exp(exponent)
            jacob = (s - x_grid) * (2.0 * s + x_grid) * (s + 2.0 * x_grid)
            raw_prob[i] = simpson_like(jacob * pade, x_grid)

        norm = simpson_like(raw_prob, order_parameter)
        probability = raw_prob / max(norm, 1e-14)
        rate_function = -(1.0 / self.N) * np.log(probability + small_value)
        return order_parameter, probability, rate_function


def normalize_nonnegative_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_pos = np.clip(y, 0.0, None)
    norm = simpson_like(y_pos, x)
    if norm <= 0.0:
        return y_pos
    return y_pos / norm


def solve_beta_epsilon(num_particles: int, mean_chi: float, mean_s_hist: float) -> tuple[float, np.ndarray, np.ndarray]:
    def mean_s_ldt(beta_epsilon: float) -> tuple[float, np.ndarray, np.ndarray]:
        model = AnalyticalMethods(num_particles=num_particles, mean_chi=mean_chi, beta_epsilon=beta_epsilon)
        s_grid, prob, _ = model.large_deviations_3D()
        s_mean = simpson_like(s_grid * prob, s_grid)
        return s_mean, s_grid, prob

    def objective(beta_epsilon: float) -> float:
        s_mean, _, _ = mean_s_ldt(beta_epsilon)
        return mean_s_hist - s_mean

    scan = np.linspace(-20.0, 20.0, 81)
    vals = np.array([objective(float(b)) for b in scan])

    lo = hi = None
    for i in range(scan.size - 1):
        if vals[i] == 0.0:
            lo = hi = float(scan[i])
            break
        if vals[i] * vals[i + 1] < 0.0:
            lo = float(scan[i])
            hi = float(scan[i + 1])
            break

    if lo is not None and hi is not None and lo != hi:
        # Bisection root solve.
        a, b = lo, hi
        fa, fb = objective(a), objective(b)
        for _ in range(80):
            c = 0.5 * (a + b)
            fc = objective(c)
            if abs(fc) < 1e-7 or abs(b - a) < 1e-6:
                a = b = c
                break
            if fa * fc <= 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        beta = 0.5 * (a + b)
    elif lo is not None and hi is not None:
        beta = lo
    else:
        # Fallback: choose best value on a dense grid.
        fine = np.linspace(-20.0, 20.0, 401)
        fvals = np.array([abs(objective(float(b))) for b in fine])
        beta = float(fine[int(np.argmin(fvals))])

    _, s_ldt, p_ldt = mean_s_ldt(beta)
    return beta, s_ldt, p_ldt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot histogram of largest eigenvalue S with CLT and LDT overlays."
    )
    parser.add_argument("--gro-dir", default=None, help="Directory containing frame*.gro")
    parser.add_argument("--s-file", default=None, help="Optional precomputed S timeseries (.npy)")
    parser.add_argument("--chi-file", default=None, help="Optional precomputed susceptibility timeseries (.npy)")
    parser.add_argument("--num-particles", type=int, default=None, help="Molecule count N (required if --gro-dir is omitted)")
    parser.add_argument("--nframes", type=int, default=1000, help="Number of frames")
    parser.add_argument("--natoms-per-mol", type=int, default=38, help="Atoms per molecule")
    parser.add_argument("--axis", choices=["molzax", "ddir"], default="molzax", help="Orientation axis")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--out", default="reports/s_isotropic_theory.png", help="Output PNG")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress")
    args = parser.parse_args()

    if args.gro_dir is None and (args.s_file is None or args.chi_file is None):
        raise SystemExit(
            "Provide --gro-dir, or provide both --s-file and --chi-file."
        )

    if args.s_file is not None:
        s_values = np.load(args.s_file)
    else:
        gro_dir = Path(args.gro_dir)
        s_values = compute_order_for_dir(
            gro_dir=gro_dir,
            nframes=args.nframes,
            natoms_per_mol=args.natoms_per_mol,
            show_progress=not args.no_progress,
            axis=args.axis,
        )

    if args.chi_file is not None:
        chi_ts = np.load(args.chi_file)
    else:
        gro_dir = Path(args.gro_dir)
        chi_ts = compute_susceptibility_timeseries(
            gro_dir=gro_dir,
            nframes=args.nframes,
            natoms_per_mol=args.natoms_per_mol,
            show_progress=not args.no_progress,
            axis=args.axis,
        )

    mean_s_hist = float(np.mean(s_values))
    mean_chi = float(np.mean(chi_ts))

    if args.num_particles is not None:
        num_particles = int(args.num_particles)
    elif args.gro_dir is not None:
        gro_dir = Path(args.gro_dir)
        first = read_gro_frame(gro_dir / "frame0.gro", natoms_per_mol=args.natoms_per_mol)
        num_particles = int(first.positions.shape[0])
    else:
        raise SystemExit("When --gro-dir is omitted, --num-particles is required.")

    beta_epsilon, s_ldt, p_ldt = solve_beta_epsilon(
        num_particles=num_particles,
        mean_chi=mean_chi,
        mean_s_hist=mean_s_hist,
    )

    model = AnalyticalMethods(num_particles=num_particles, mean_chi=mean_chi, beta_epsilon=beta_epsilon)
    s_clt = np.linspace(0.0, 0.5, 600)
    p_clt = normalize_nonnegative_pdf(s_clt, model.central_limit_3D(s_clt))

    style_path = Path(__file__).with_name("prl.mplstyle")
    if style_path.exists():
        plt.style.use(style_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.hist(s_values, bins=args.bins, density=True, alpha=0.35, color="0.4", label="Histogram")
    ax.plot(s_clt, p_clt, color="tab:blue", lw=2.0, label="CLT")
    ax.plot(s_ldt, p_ldt, color="tab:red", lw=2.0, label="LDT")
    ax.axvline(mean_s_hist, color="black", lw=1.4, ls="--", label=f"mean(S)={mean_s_hist:.4f}")

    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel("Largest eigenvalue S")
    ax.set_ylabel("Probability density")
    title_label = Path(args.gro_dir).name if args.gro_dir else "precomputed"
    ax.set_title(f"S distribution with CLT/LDT overlays ({title_label})")
    ax.legend(frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)

    print(f"Wrote {out}")
    print(f"N molecules: {num_particles}")
    print(f"mean chi: {mean_chi:.6f}")
    print(f"mean S (hist): {mean_s_hist:.6f}")
    print(f"beta*epsilon (fit): {beta_epsilon:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
