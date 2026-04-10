"""Reproduce angular_steering.ipynb as a standalone script.

Matches the notebook exactly:
  - TransformerLens for both extraction and generation
  - Hooks on resid_pre + resid_mid (residual stream, not layernorm outputs)
  - No KV cache (hooks fire on all positions each step)
  - Batched generation with padding
  - Per-sample normalized directions, max_sim layer selection

Usage:
    conda run -n MA4198 python run_notebook_as_script.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output-dir output/notebook_repro \
        --angle-step 10 \
        --num-test-samples 104
"""

import argparse
import functools
import gc
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization (matches notebook cell 14)
# ---------------------------------------------------------------------------

def instructions_to_chat_tokens(tokenizer, instructions):
    if tokenizer.chat_template:
        convos = [[{"role": "user", "content": inst}] for inst in instructions]
        return tokenizer.apply_chat_template(
            convos, padding=True, truncation=False,
            add_generation_prompt=True, return_tensors="pt",
        )
    return tokenizer(
        instructions, padding=True, truncation=False, return_tensors="pt",
    ).input_ids


# ---------------------------------------------------------------------------
# Activation extraction (matches notebook cells 21-25)
# ---------------------------------------------------------------------------

def get_template_suffix_toks(tokenizer):
    toks = instructions_to_chat_tokens(tokenizer, ["a", "b"])
    suffix = toks[0]
    for i in range(len(toks[0]) - 1, -1, -1):
        if toks[0][i] != toks[1][i]:
            suffix = toks[0][i + 1:]
            break
    return tokenizer.convert_ids_to_tokens(suffix)


def extract_activations(model, instructions, batch_size, act_names, num_last_tokens):
    """Returns (num_layers, num_act_modules, num_samples, num_last_tokens, d_model)."""
    toks = instructions_to_chat_tokens(model.tokenizer, instructions)
    cache_accum = {}

    with torch.no_grad():
        for i in range(0, len(instructions), batch_size):
            batch = toks[i:i + batch_size].to(model.cfg.device)
            _, batch_cache = model.run_with_cache(
                batch, names_filter=lambda name: "resid" in name,
                return_cache_object=False,
            )
            for k, v in batch_cache.items():
                v_cpu = v[:, -num_last_tokens:, :].cpu()
                if k not in cache_accum:
                    cache_accum[k] = v_cpu
                else:
                    cache_accum[k] = torch.cat([cache_accum[k], v_cpu], dim=0)
            del batch_cache
            if (i + batch_size) % 100 == 0 or i + batch_size >= len(instructions):
                logger.info(f"  {min(i + batch_size, len(instructions))}/{len(instructions)}")

    acts = torch.stack([
        torch.stack([cache_accum[f"blocks.{layer}.hook_{act}"] for act in act_names])
        for layer in range(model.cfg.n_layers)
    ])
    return acts


# ---------------------------------------------------------------------------
# Direction computation (matches notebook cells 27, 41, 46, 51)
# ---------------------------------------------------------------------------

def compute_directions(harmful_acts, harmless_acts):
    chosen_token = -1
    ha = harmful_acts[:, :, :, chosen_token, :]
    hn = harmless_acts[:, :, :, chosen_token, :]

    ha_normed = ha / ha.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    hn_normed = hn / hn.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    ha_mean = ha_normed.mean(dim=2)
    hn_mean = hn_normed.mean(dim=2)

    ha_mean_normed = normalize(ha_mean, dim=-1)
    hn_mean_normed = normalize(hn_mean, dim=-1)

    refusal_dirs = ha_mean_normed - hn_mean_normed
    refusal_dirs = normalize(refusal_dirs, dim=-1)

    # Max-sim criterion
    dirs_flat = refusal_dirs.reshape(-1, refusal_dirs.shape[-1])
    dirs_flat_normed = normalize(dirs_flat, dim=-1)
    sim_matrix = dirs_flat_normed @ dirs_flat_normed.T
    mean_cosine = sim_matrix.mean(dim=-1)

    best_sim_idx = mean_cosine.argmax().item()
    num_act = refusal_dirs.shape[1]
    max_sim_layer = best_sim_idx // num_act
    max_sim_act_idx = best_sim_idx % num_act

    # Max-norm criterion
    raw_dirs = ha_mean_normed - hn_mean_normed
    norms = raw_dirs.reshape(-1, raw_dirs.shape[-1]).norm(dim=-1)
    best_norm_idx = norms[:-1].argmax().item()
    max_norm_layer = best_norm_idx // num_act
    max_norm_act_idx = best_norm_idx % num_act

    # PCA
    pca = PCA()
    pca.fit(dirs_flat.cpu().float().numpy())
    components = pca.components_

    return (
        refusal_dirs.cpu().float().numpy(), components,
        max_sim_layer, max_sim_act_idx,
        max_norm_layer, max_norm_act_idx,
    )


# ---------------------------------------------------------------------------
# Generation (matches notebook cells 17, 81)
# ---------------------------------------------------------------------------

def get_rotate_to_target_func(target_degree, basis1, basis2):
    """Matches notebook cell 81."""
    u = basis1 / np.linalg.norm(basis1)
    v = basis2 - (basis2 @ u) * u
    v /= np.linalg.norm(v)

    theta = np.deg2rad(target_degree)
    P = np.outer(u, u) + np.outer(v, v)
    R_theta = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    uv = np.column_stack([u, v])
    rotated_component = uv @ R_theta @ np.array([1, 0])

    P_t = torch.tensor(P, dtype=torch.float32)
    rc_t = torch.tensor(rotated_component, dtype=torch.float32)

    def func(x: Tensor):
        Px = x @ P_t.to(device=x.device, dtype=x.dtype)
        scale = Px.norm(dim=-1, keepdim=True)
        return x - Px + scale * rc_t.to(device=x.device, dtype=x.dtype)

    return func


def activation_rotation_hook(activation, hook, transformation_func):
    """Matches notebook cell 81."""
    return transformation_func(activation)


def generate_with_hooks(model, toks, max_tokens_generated, fwd_hooks):
    """Matches notebook cell 17 — no KV cache, hooks on all positions."""
    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated),
        dtype=torch.long, device=toks.device,
    )
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_toks[:, -max_tokens_generated + i] = next_tokens

    return model.tokenizer.batch_decode(
        all_toks[:, toks.shape[1]:], skip_special_tokens=True,
    )


def get_generations(model, instructions, tokenizer, fwd_hooks,
                    max_tokens_generated=512, batch_size=4):
    """Matches notebook cell 17."""
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        toks = instructions_to_chat_tokens(
            tokenizer, instructions[i:i + batch_size],
        )
        with torch.no_grad():
            gen = generate_with_hooks(
                model, toks.to(model.cfg.device),
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
            )
        # Strip after EOS
        gen = [g.split(tokenizer.eos_token)[0] for g in gen]
        generations.extend(gen)
    return generations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="output/notebook_repro")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--n-samples", type=int, default=512,
                        help="Number of training samples for direction extraction")
    parser.add_argument("--num-test-samples", type=int, default=104,
                        help="Number of test samples for generation (104=full, 4=quick)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation (keep small, no KV cache)")
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--angle-step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_name = args.model
    model_short = model_name.split("/")[-1]
    output_path = Path(args.output_dir) / model_short
    output_path.mkdir(parents=True, exist_ok=True)

    if model_name not in OFFICIAL_MODEL_NAMES:
        OFFICIAL_MODEL_NAMES.append(model_name)

    # --- Load model ---
    logger.info(f"Loading {model_name} via TransformerLens...")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    if not model.tokenizer.pad_token:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    template_suffix_toks = get_template_suffix_toks(model.tokenizer)
    num_last_tokens = len(template_suffix_toks) if template_suffix_toks else 1
    logger.info(f"Template suffix ({num_last_tokens} toks): {template_suffix_toks}")

    act_names = ["resid_mid", "resid_post"]

    # --- Load data ---
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_activation_control.utils import get_harmful_instructions, get_harmless_instructions

    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions()
    harmful_train = harmful_train[:args.n_samples]
    harmless_train = harmless_train[:args.n_samples]
    test_samples = harmful_test[:args.num_test_samples]

    logger.info(f"Train: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    logger.info(f"Test: {len(test_samples)} samples")

    # --- Extract directions ---
    dirs_file = output_path / "refusal_dirs.npy"
    if dirs_file.exists():
        logger.info(f"Loading cached directions: {dirs_file}")
        refusal_dirs = np.load(dirs_file)
        components = np.load(output_path / "pca_components.npy")
        meta = json.loads((output_path / "extraction_meta.json").read_text())
        max_sim_layer = meta["max_sim_layer"]
        max_sim_act_idx = meta["max_sim_act_idx"]
    else:
        logger.info("Extracting harmful activations...")
        harmful_acts = extract_activations(
            model, harmful_train, args.extraction_batch_size, act_names, num_last_tokens,
        )
        gc.collect(); torch.cuda.empty_cache()

        logger.info("Extracting harmless activations...")
        harmless_acts = extract_activations(
            model, harmless_train, args.extraction_batch_size, act_names, num_last_tokens,
        )
        gc.collect(); torch.cuda.empty_cache()

        logger.info("Computing directions...")
        refusal_dirs, components, max_sim_layer, max_sim_act_idx, max_norm_layer, max_norm_act_idx = \
            compute_directions(harmful_acts, harmless_acts)

        np.save(dirs_file, refusal_dirs)
        np.save(output_path / "pca_components.npy", components)
        meta = {
            "max_sim_layer": max_sim_layer, "max_sim_act_idx": max_sim_act_idx,
            "max_norm_layer": max_norm_layer, "max_norm_act_idx": max_norm_act_idx,
        }
        (output_path / "extraction_meta.json").write_text(json.dumps(meta, indent=2))

    chosen_layer = max_sim_layer
    chosen_act_idx = max_sim_act_idx
    refusal_dir = refusal_dirs[chosen_layer][chosen_act_idx]
    logger.info(f"Chosen: layer {chosen_layer}, act {act_names[chosen_act_idx]}")

    # --- Generate at each angle ---
    responses_file = output_path / "steered_responses.json"
    if responses_file.exists():
        logger.info(f"Loading cached responses: {responses_file}")
        all_responses = json.loads(responses_file.read_text())
    else:
        all_responses = {}

    num_layers = model.cfg.n_layers
    intervention_layers = list(range(num_layers))

    for degree in range(0, 360, args.angle_step):
        if str(degree) in all_responses:
            logger.info(f"  {degree}°: cached ({len(all_responses[str(degree)])} samples)")
            continue

        logger.info(f"  Generating at {degree}°...")
        transformation_func = get_rotate_to_target_func(
            target_degree=degree,
            basis1=refusal_dir.copy(),
            basis2=components[0].copy(),
        )

        fwd_hooks = [
            (
                f"blocks.{layer}.hook_{act_name}",
                functools.partial(activation_rotation_hook, transformation_func=transformation_func),
            )
            for layer in intervention_layers
            for act_name in ["resid_pre", "resid_mid"]
        ]

        responses = get_generations(
            model, test_samples, model.tokenizer,
            fwd_hooks=fwd_hooks,
            max_tokens_generated=args.max_tokens,
            batch_size=args.batch_size,
        )
        all_responses[str(degree)] = responses

        # Save after each angle (resumable)
        responses_file.write_text(json.dumps(all_responses, indent=2))
        logger.info(f"  {degree}°: saved {len(responses)} responses")

    logger.info(f"Done. Responses saved to {responses_file}")


if __name__ == "__main__":
    main()
