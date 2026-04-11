"""
Filter a steering config .npy to keep only a subset of layer modules.

The default sentiment extraction saves the direction at every layer of the
model (e.g. 71 module entries for a 36-layer Qwen2.5-3B). When angular
steering applies at every layer, the rotation accumulates through the
network and can destroy outputs even when the direction itself is correct
(3B max_sim_27_mid CAA works but angular fails).

This tool loads a full config, keeps only the modules corresponding to a
specified set of transformer blocks, and saves a new config file.

Usage:
    # Keep only layer 27 modules
    python filter_config_layers.py \\
        --input output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid-pca_0.npy \\
        --layers 27 \\
        --output output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid_L27only-pca_0.npy

    # Keep layers 25-29 (top-5 by cosine similarity)
    python filter_config_layers.py \\
        --input output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid-pca_0.npy \\
        --layers 25,26,27,28,29 \\
        --output output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid_L25-29-pca_0.npy
"""

import argparse
import re
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to full config .npy")
    parser.add_argument("--output", required=True, help="Path to write filtered config")
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices to keep (e.g., '27' or '25,26,27,28,29')",
    )
    args = parser.parse_args()

    layers_to_keep = set(int(x) for x in args.layers.split(","))
    print(f"Layers to keep: {sorted(layers_to_keep)}")

    config = np.load(args.input, allow_pickle=True).item()
    print(f"Loaded config with {len(config)} module entries")

    # Parse "model.layers.<N>.<module_type>" from each key, keep only those
    # whose layer index N is in layers_to_keep.
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    filtered = {}
    kept_layers = set()
    for module_name, directions in config.items():
        m = pattern.match(module_name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        if layer_idx in layers_to_keep:
            filtered[module_name] = directions
            kept_layers.add(layer_idx)

    print(f"Filtered to {len(filtered)} module entries across layers {sorted(kept_layers)}")
    print(f"Modules kept:")
    for k in sorted(filtered.keys()):
        print(f"  {k}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, filtered, allow_pickle=True)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
