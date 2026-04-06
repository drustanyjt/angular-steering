"""Extract steering directions using TransformerLens, matching the notebook exactly.

Usage:
    conda run -n MA4198 python extract_directions_tl.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output-dir output \
        --language en
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.nn.functional import normalize
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

from llm_activation_control.utils import (
    get_harmful_instructions,
    get_harmless_instructions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation (matches notebook Cell 14)
# ---------------------------------------------------------------------------

def instructions_to_chat_tokens(tokenizer, instructions):
    if tokenizer.chat_template:
        convos = [[{"role": "user", "content": inst}] for inst in instructions]
        return tokenizer.apply_chat_template(
            convos,
            padding=True,
            truncation=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(
        instructions, padding=True, truncation=False, return_tensors="pt"
    ).input_ids


def get_template_suffix_toks(tokenizer):
    """Find tokens common to all samples at the end (the fixed chat template suffix)."""
    toks = instructions_to_chat_tokens(tokenizer, ["a", "b"])
    suffix = toks[0]
    for i in range(len(toks[0]) - 1, -1, -1):
        if toks[0][i] != toks[1][i]:
            suffix = toks[0][i + 1:]
            break
    return tokenizer.convert_ids_to_tokens(suffix)


# ---------------------------------------------------------------------------
# Activation extraction (matches notebook Cell 21 / 23)
# ---------------------------------------------------------------------------

def extract_activations(model, instructions, batch_size, act_names, num_last_tokens):
    """Returns tensor of shape (num_layers, num_act_modules, num_samples, num_last_tokens, d_model)."""
    cache_accum = {}

    toks = instructions_to_chat_tokens(model.tokenizer, instructions)
    n = len(instructions)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = toks[i: i + batch_size].to(model.cfg.device)
            _, batch_cache = model.run_with_cache(
                batch,
                names_filter=lambda name: "resid" in name,
                return_cache_object=False,
            )
            for k, v in batch_cache.items():
                # v: (batch, seq_len, d_model) — keep only last num_last_tokens
                v_cpu = v[:, -num_last_tokens:, :].cpu()
                if k not in cache_accum:
                    cache_accum[k] = v_cpu
                else:
                    cache_accum[k] = torch.cat([cache_accum[k], v_cpu], dim=0)
            del batch_cache
            logger.info(f"  {min(i + batch_size, n)}/{n}")

    num_layers = model.cfg.n_layers
    # stack: (num_layers, num_act_modules, num_samples, num_last_tokens, d_model)
    acts = torch.stack([
        torch.stack([cache_accum[f"blocks.{layer}.hook_{act}"] for act in act_names])
        for layer in range(num_layers)
    ])
    return acts  # (L, A, N, T, D)


# ---------------------------------------------------------------------------
# Direction computation (matches notebook Cells 25, 27, 41, 51)
# ---------------------------------------------------------------------------

def compute_directions(harmful_acts, harmless_acts):
    """
    harmful_acts, harmless_acts: (L, A, N, T, D)
    Returns refusal_dirs: (L, A, D), pca_components: (D, D),
            max_sim_layer, max_sim_act_idx, max_norm_layer, max_norm_act_idx
    """
    chosen_token = -1  # last template suffix token

    # Normalise each activation sample, then take mean and renormalise
    # shape: (L, A, N, D)
    ha = harmful_acts[:, :, :, chosen_token, :]
    hn = harmless_acts[:, :, :, chosen_token, :]

    ha_normed = ha / ha.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    hn_normed = hn / hn.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    ha_mean = ha_normed.mean(dim=2)   # (L, A, D)
    hn_mean = hn_normed.mean(dim=2)

    ha_mean_normed = normalize(ha_mean, dim=-1)
    hn_mean_normed = normalize(hn_mean, dim=-1)

    refusal_dirs = ha_mean_normed - hn_mean_normed  # (L, A, D)
    refusal_dirs = normalize(refusal_dirs, dim=-1)

    # ---- max_sim criterion (highest mean pairwise cosine similarity) ----
    dirs_flat = refusal_dirs.reshape(-1, refusal_dirs.shape[-1])  # (L*A, D)
    dirs_flat_normed = normalize(dirs_flat, dim=-1)
    sim_matrix = dirs_flat_normed @ dirs_flat_normed.T  # (L*A, L*A)
    mean_cosine = sim_matrix.mean(dim=-1)  # (L*A,)

    best_sim_idx = mean_cosine.argmax().item()
    num_act = refusal_dirs.shape[1]
    max_sim_layer = best_sim_idx // num_act
    max_sim_act_idx = best_sim_idx % num_act

    logger.info(f"Max-sim best layer: {max_sim_layer}, act_idx: {max_sim_act_idx}")

    # ---- max_norm criterion ----
    raw_dirs = ha_mean_normed - hn_mean_normed  # before normalising
    norms = raw_dirs.reshape(-1, raw_dirs.shape[-1]).norm(dim=-1)
    # exclude last entry (notebook uses [:-1])
    best_norm_idx = norms[:-1].argmax().item()
    max_norm_layer = best_norm_idx // num_act
    max_norm_act_idx = best_norm_idx % num_act

    logger.info(f"Max-norm best layer: {max_norm_layer}, act_idx: {max_norm_act_idx}")

    # ---- PCA on all candidate directions ----
    pca = PCA()
    pca.fit(dirs_flat.cpu().float().numpy())
    components = pca.components_  # (D, D)

    return (
        refusal_dirs.cpu().float().numpy(),
        components,
        max_sim_layer, max_sim_act_idx,
        max_norm_layer, max_norm_act_idx,
    )


# ---------------------------------------------------------------------------
# Save steering config (matches notebook Cell 76)
# ---------------------------------------------------------------------------

def save_steering_configs(
    refusal_dirs, components, model_path, output_path,
    max_sim_layer, max_sim_act_idx,
    max_norm_layer, max_norm_act_idx,
    act_names, language,
):
    target_modules_names = ["mid", "post"]
    layernorm_modules = ["input_layernorm", "post_attention_layernorm"]

    mean_d = refusal_dirs.reshape(-1, refusal_dirs.shape[-1]).mean(axis=0)
    mean_d /= np.linalg.norm(mean_d)

    num_layers = refusal_dirs.shape[0]

    for first_direction, first_dir_name in [
        (
            refusal_dirs[max_norm_layer][max_norm_act_idx].copy(),
            f"dir_max_norm_{max_norm_layer}_{target_modules_names[max_norm_act_idx]}",
        ),
        (
            refusal_dirs[max_sim_layer][max_sim_act_idx].copy(),
            f"dir_max_sim_{max_sim_layer}_{target_modules_names[max_sim_act_idx]}",
        ),
        (mean_d.copy(), "dir_mean"),
    ]:
        second_direction = components[0].copy()

        steering_config = {}
        for layer_idx in range(num_layers):
            for module in layernorm_modules:
                if module != "input_layernorm":
                    module_name = f"model.layers.{layer_idx}.{module}"
                elif layer_idx < num_layers - 1:
                    module_name = f"model.layers.{layer_idx + 1}.{module}"
                else:
                    continue
                steering_config[module_name] = {
                    "mode": "rotate_to",
                    "first_direction": first_direction,
                    "second_direction": second_direction,
                }

        fname = f"steering_config-{language}-{first_dir_name}-pca_0.npy"
        np.save(output_path / fname, steering_config)
        logger.info(f"Saved: {fname}")

    logger.info(f"Done. Configs saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir",  type=str, default="./output")
    parser.add_argument("--language",    type=str, default="en")
    parser.add_argument("--n-samples",   type=int, default=512)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model via TransformerLens
    if args.model not in OFFICIAL_MODEL_NAMES:
        OFFICIAL_MODEL_NAMES.append(args.model)

    logger.info(f"Loading {args.model} via TransformerLens...")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    model.eval()

    if not model.tokenizer.pad_token:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # Template suffix tokens
    template_suffix_toks = get_template_suffix_toks(model.tokenizer)
    if not template_suffix_toks:
        template_suffix_toks = ["<last token>"]
    num_last_tokens = len(template_suffix_toks)
    logger.info(f"Template suffix ({num_last_tokens} toks): {template_suffix_toks}")

    act_names = ["resid_mid", "resid_post"]

    # Load data
    logger.info("Loading datasets...")
    harmful_train, _ = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions()
    harmful_train  = harmful_train[:args.n_samples]
    harmless_train = harmless_train[:args.n_samples]
    logger.info(f"Using {len(harmful_train)} harmful, {len(harmless_train)} harmless samples")

    # Extract activations
    logger.info("Extracting harmful activations...")
    harmful_acts = extract_activations(
        model, harmful_train, args.batch_size, act_names, num_last_tokens
    )
    gc.collect(); torch.cuda.empty_cache()

    logger.info("Extracting harmless activations...")
    harmless_acts = extract_activations(
        model, harmless_train, args.batch_size, act_names, num_last_tokens
    )
    gc.collect(); torch.cuda.empty_cache()

    # Compute directions
    logger.info("Computing steering directions...")
    refusal_dirs, components, max_sim_layer, max_sim_act_idx, max_norm_layer, max_norm_act_idx = \
        compute_directions(harmful_acts, harmless_acts)

    # Save configs
    save_steering_configs(
        refusal_dirs, components, args.model, output_path,
        max_sim_layer, max_sim_act_idx,
        max_norm_layer, max_norm_act_idx,
        act_names, args.language,
    )


if __name__ == "__main__":
    main()
