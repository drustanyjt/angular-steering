"""Generate layer sweep plots using matplotlib (PDF-friendly for LaTeX).

Usage:
    python pytorch_pure/plot_layer_sweep_mpl.py \
        --csv output/Qwen2.5-3B-Instruct/layer_sweep_results.csv \
        --output-dir figures/layer_sweep
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

METRICS = ["substring_matching", "harmbench", "llamaguard3"]
METRIC_LABELS = {
    "substring_matching": "Substring matching (refusal ↓)",
    "harmbench": "HarmBench (harmful ↑)",
    "llamaguard3": "LlamaGuard3 (unsafe ↑)",
}
COLOURS = {
    "substring_matching": "#2563eb",   # blue
    "harmbench": "#dc2626",            # red
    "llamaguard3": "#16a34a",          # green
}
CONDITIONS = ["per_layer_single", "fixed_single"]
CONDITION_LABELS = {
    "per_layer_single": "Per-layer direction,\nsingle-layer ablation",
    "fixed_single": "Fixed direction,\nsingle-layer ablation",
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sweep = df[df["layer"] != -1].copy()
    sweep["layer"] = sweep["layer"].astype(int)
    fixed_all = df[df["layer"] == -1].copy()
    return sweep, fixed_all


def make_line_plot(sweep: pd.DataFrame, fixed_all: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        sub = sweep[sweep["condition"] == condition].sort_values("layer")

        for metric in METRICS:
            ax.plot(sub["layer"], sub[metric],
                    color=COLOURS[metric], linewidth=1.8,
                    label=METRIC_LABELS[metric])

        # fixed_all as horizontal dashed reference
        if not fixed_all.empty:
            for metric in METRICS:
                val = float(fixed_all[metric].iloc[0])
                ax.axhline(val, color=COLOURS[metric], linestyle="--",
                           linewidth=1.2, alpha=0.7)

        ax.set_title(CONDITION_LABELS[condition], fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xlim(sub["layer"].min(), sub["layer"].max())
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score", fontsize=10)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Add dashed line entry for fixed_all reference
    import matplotlib.lines as mlines
    dash = mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.2,
                         label="Fixed-direction all-layer (reference)")
    fig.legend(handles + [dash], labels + [dash.get_label()],
               loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Layer sweep: directional ablation on Qwen2.5-3B-Instruct",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "layer_sweep_line.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "layer_sweep_line.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved line plot → {output_dir / 'layer_sweep_line.pdf'}")


def make_polar_plot(sweep: pd.DataFrame, fixed_all: pd.DataFrame,
                    output_dir: Path, num_layers: int):
    """Polar plot where angle = layer index mapped linearly to [0, 2π)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             subplot_kw={"projection": "polar"})

    for ax, condition in zip(axes, CONDITIONS):
        sub = sweep[sweep["condition"] == condition].sort_values("layer")
        layers = sub["layer"].values
        # Map layers to angles (0 at top, clockwise)
        angles = 2 * np.pi * layers / num_layers
        # Close the loop
        angles_c = np.append(angles, angles[0])

        for metric in METRICS:
            vals = sub[metric].values
            vals_c = np.append(vals, vals[0])
            ax.plot(angles_c, vals_c, color=COLOURS[metric],
                    linewidth=1.8, label=METRIC_LABELS[metric])

        # fixed_all reference as filled circle
        if not fixed_all.empty:
            for metric in METRICS:
                val = float(fixed_all[metric].iloc[0])
                theta = np.linspace(0, 2 * np.pi, 200)
                ax.plot(theta, [val] * 200, color=COLOURS[metric],
                        linestyle="--", linewidth=1.0, alpha=0.6)

        ax.set_title(CONDITION_LABELS[condition], fontsize=10, pad=14)
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise

        # Label every 4 layers (~every 10% of total)
        step = max(1, num_layers // 12)
        tick_layers = list(range(0, num_layers, step))
        tick_angles = [2 * np.pi * l / num_layers for l in tick_layers]
        ax.set_xticks(tick_angles)
        ax.set_xticklabels([str(l) for l in tick_layers], fontsize=8)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    import matplotlib.lines as mlines
    dash = mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.0,
                         label="Fixed-direction all-layer (reference)")
    fig.legend(handles + [dash], labels + [dash.get_label()],
               loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Layer sweep (polar): directional ablation on Qwen2.5-3B-Instruct",
                 fontsize=12)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "layer_sweep_polar.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "layer_sweep_polar.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved polar plot → {output_dir / 'layer_sweep_polar.pdf'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("figures/layer_sweep"), type=Path)
    args = parser.parse_args()

    sweep, fixed_all = load(args.csv)
    num_layers = int(sweep["layer"].max()) + 1

    make_line_plot(sweep, fixed_all, args.output_dir)
    make_polar_plot(sweep, fixed_all, args.output_dir, num_layers)


if __name__ == "__main__":
    main()
