from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np

import matplotlib

# Force non-interactive backend to avoid GUI hangs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_s_values(order_dir: Path) -> list[tuple[str, np.ndarray]]:
    files = sorted(order_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {order_dir}")
    data: list[tuple[str, np.ndarray]] = []
    for path in files:
        data.append((path.stem, np.load(path)))
    return data


def plot_avg_s_vs_temperature(
    order_dir: Path,
    out_path: Path,
    bins: int,
) -> None:
    s_datasets = load_s_values(order_dir)
    style_path = Path(__file__).with_name("prl.mplstyle")
    if style_path.exists():
        plt.style.use(style_path)

    # Map file stems to temperatures (K)
    temp_map = {
        "gro_S": 356.0,
        "gro_2_S": 366.0,
        "gro_3_S": 376.0,
        "gro_4_S": 386.0,
    }

    temps = []
    means = []
    stds = []
    for label, s_values in s_datasets:
        if label not in temp_map:
            continue
        temps.append(temp_map[label])
        means.append(float(np.mean(s_values)))
        stds.append(float(np.std(s_values, ddof=1)))

    if not temps:
        raise ValueError("No datasets matched the temperature mapping.")

    order = np.argsort(temps)
    temps = np.array(temps)[order]
    means = np.array(means)[order]
    stds = np.array(stds)[order]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(temps, means, yerr=stds, fmt="o-", color="black", capsize=3)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Average S")
    ax.set_title("Average nematic order parameter vs temperature")

    for t, m in zip(temps, means):
        ax.annotate(
            f"({m:.3f}, {t:.0f}K)",
            xy=(t, m),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot P(S) from largest PAF eigenvalue across trajectories."
    )
    parser.add_argument(
        "--order-dir",
        default="data/order",
        help="Directory containing saved S .npy files",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bins",
    )
    parser.add_argument(
        "--out",
        default="reports/p_s_hist.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--avg-out",
        default="reports/avg_s_vs_t.png",
        help="Output PNG path for average S vs temperature plot",
    )
    parser.add_argument(
        "--avg-only",
        action="store_true",
        help="Only generate the average S vs temperature plot",
    )
    args = parser.parse_args()

    order_dir = Path(args.order_dir)
    s_datasets = load_s_values(order_dir)

    style_path = Path(__file__).with_name("prl.mplstyle")
    if style_path.exists():
        plt.style.use(style_path)

    if args.avg_only:
        plot_avg_s_vs_temperature(
            order_dir=order_dir,
            out_path=Path(args.avg_out),
            bins=args.bins,
        )
        return 0

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    all_values = np.concatenate([vals for _, vals in s_datasets])
    x_min = float(np.min(all_values))
    x_max = float(np.max(all_values))
    bins = np.linspace(x_min, x_max, args.bins + 1)

    color_map = {
        "gro_S": ("356K", "blue"),
        "gro_2_S": ("366K", "red"),
        "gro_3_S": ("376K", "green"),
        "gro_4_S": ("386K", "magenta"),
    }

    max_density = 0.0
    for label, s_values in s_datasets:
        hist, _ = np.histogram(s_values, bins=bins, density=True)
        max_density = max(max_density, float(np.max(hist)))

        temp_label, color = color_map.get(label, (label, "black"))
        ax.hist(
            s_values,
            bins=bins,
            density=True,
            linewidth=1.8,
            color=color,
            label=temp_label,
        )

    ax.set_xlabel("S (largest eigenvalue)")
    ax.set_ylabel("P(S)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, max_density * 1.05)
    ax.legend(frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Nematic order parameter distribution", fontsize=14)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")

    plot_avg_s_vs_temperature(
        order_dir=order_dir,
        out_path=Path(args.avg_out),
        bins=args.bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
