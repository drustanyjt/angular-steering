"""Plot layer sweep results from layer_sweep_results.csv.

Produces two figures:
  1. Line plot of each metric vs layer, one subplot per condition.
  2. Same data as a polar plot (angle = layer index mapped to 0-360°),
     matching the visual style of the steering evaluation figures.

Usage:
    python pytorch_pure/plot_layer_sweep.py \
        --csv output/Qwen2.5-3B-Instruct/layer_sweep_results.csv \
        --output-dir figures/layer_sweep
"""
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = ["substring_matching", "harmbench", "llamaguard3"]
METRIC_LABELS = {
    "substring_matching": "Substring matching (refusal ↓)",
    "harmbench": "HarmBench (harmful ↑)",
    "llamaguard3": "LlamaGuard3 (unsafe ↑)",
}
COLOUR_MAP = {
    "substring_matching": "blue",
    "harmbench": "red",
    "llamaguard3": "green",
}
CONDITIONS = ["per_layer_single", "fixed_single", "fixed_all"]
CONDITION_LABELS = {
    "per_layer_single": "Per-layer direction, single-layer ablation",
    "fixed_single": "Fixed direction, single-layer ablation",
    "fixed_all": "Fixed direction, all-layer ablation",
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def make_line_plot(df: pd.DataFrame, output_dir: Path):
    """Line plot: metric value vs layer, one column per condition."""
    sweep_conditions = [c for c in CONDITIONS if c != "fixed_all"]
    fig = make_subplots(
        rows=1,
        cols=len(sweep_conditions),
        subplot_titles=[CONDITION_LABELS[c] for c in sweep_conditions],
        shared_yaxes=True,
    )

    for col_idx, condition in enumerate(sweep_conditions, start=1):
        sub = df[df["condition"] == condition].copy()
        sub = sub[sub["layer"] != "all"].copy()
        sub["layer"] = sub["layer"].astype(int)
        sub = sub.sort_values("layer")

        for metric in METRICS:
            fig.add_trace(
                go.Scatter(
                    x=sub["layer"],
                    y=sub[metric],
                    name=METRIC_LABELS[metric],
                    line=dict(color=COLOUR_MAP[metric], width=2),
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )

        # Add fixed_all as horizontal reference lines
        fixed_all = df[df["condition"] == "fixed_all"]
        if not fixed_all.empty:
            for metric in METRICS:
                val = float(fixed_all[metric].iloc[0])
                fig.add_hline(
                    y=val,
                    line=dict(color=COLOUR_MAP[metric], dash="dot", width=1),
                    row=1,
                    col=col_idx,
                    annotation_text=f"{metric} (all-layer)" if col_idx == 1 else "",
                    annotation_position="bottom right",
                )

    fig.update_layout(
        title="Layer sweep: directional ablation on Qwen2.5-3B-Instruct",
        yaxis_title="Score",
        xaxis_title="Layer",
        height=450,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_dir / "layer_sweep_line.pdf"))
    fig.write_html(str(output_dir / "layer_sweep_line.html"))
    print(f"Saved line plot to {output_dir}")


def make_polar_plot(df: pd.DataFrame, output_dir: Path, num_layers: int):
    """Polar plot: layer index mapped linearly to 0–360°.

    Each metric is a radial trace; the angle encodes the layer number.
    Matches the visual style of the angular steering evaluation figures.
    """
    sweep_conditions = [c for c in CONDITIONS if c != "fixed_all"]
    fig = make_subplots(
        rows=1,
        cols=len(sweep_conditions),
        specs=[[{"type": "polar"}] * len(sweep_conditions)],
        subplot_titles=[CONDITION_LABELS[c] for c in sweep_conditions],
    )

    # Map layer index → angle in degrees
    def layer_to_angle(layer: int) -> float:
        return (layer / num_layers) * 360.0

    for col_idx, condition in enumerate(sweep_conditions, start=1):
        sub = df[df["condition"] == condition].copy()
        sub = sub[sub["layer"] != "all"].copy()
        sub["layer"] = sub["layer"].astype(int)
        sub = sub.sort_values("layer")

        angles = [layer_to_angle(l) for l in sub["layer"]]
        # Close the loop
        angles_closed = angles + [angles[0]]

        for metric in METRICS:
            values = list(sub[metric])
            values_closed = values + [values[0]]
            polar_kw = dict(polar=f"polar{col_idx}" if col_idx > 1 else "polar")
            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=angles_closed,
                    name=METRIC_LABELS[metric],
                    line=dict(color=COLOUR_MAP[metric], width=2),
                    mode="lines",
                    showlegend=(col_idx == 1),
                    **polar_kw,
                ),
                row=1,
                col=col_idx,
            )

    # Tick every ~10 layers
    tick_step = max(1, num_layers // 36)
    tickvals = list(range(0, num_layers, tick_step))
    ticktext = [str(l) for l in tickvals]
    tick_angles = [layer_to_angle(l) for l in tickvals]

    polar_axes = {}
    for col_idx in range(1, len(sweep_conditions) + 1):
        key = "polar" if col_idx == 1 else f"polar{col_idx}"
        polar_axes[key] = dict(
            angularaxis=dict(
                tickvals=tick_angles,
                ticktext=ticktext,
                tickfont=dict(size=10),
                direction="clockwise",
                rotation=90,
            ),
            radialaxis=dict(range=[0, 1]),
        )

    fig.update_layout(
        title="Layer sweep (polar): directional ablation on Qwen2.5-3B-Instruct",
        height=500,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        **polar_axes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_dir / "layer_sweep_polar.pdf"))
    fig.write_html(str(output_dir / "layer_sweep_polar.html"))
    print(f"Saved polar plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("figures/layer_sweep"), type=Path)
    args = parser.parse_args()

    df = load_csv(args.csv)

    # Infer number of layers from the data
    layer_vals = df[df["layer"] != "all"]["layer"].astype(int)
    num_layers = int(layer_vals.max()) + 1

    make_line_plot(df, args.output_dir)
    make_polar_plot(df, args.output_dir, num_layers)


if __name__ == "__main__":
    main()
