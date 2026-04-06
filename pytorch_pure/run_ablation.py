"""Generate responses with directional ablation (pure PyTorch).

Metric convention matches the paper:
  - substring_matching : refusal rate  (↓ lower = fewer refusals = stronger attack)
  - harmbench          : harmful rate  (↑ higher = more harmful content)
  - llamaguard3        : unsafe rate   (↑ higher = more unsafe content)
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_pure.utils import add_hooks, get_input_data, tokenize_instructions_fn
from evaluate_jailbreak import evaluate_jailbreak

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def to_paper_convention(scores: list, method: str) -> float:
    """Convert raw evaluate_jailbreak scores to the paper's display convention.

    evaluate_jailbreak returns ASR for every method:
        1 = jailbreak succeeded  (response is harmful / no refusal phrase)
        0 = refusal              (response is safe)

    Paper convention:
        harmbench    ↑  = harmful rate  → same as ASR, no flip
        llamaguard3  ↑  = unsafe rate   → same as ASR, no flip
        substring    ↓  = refusal rate  → FLIP (1 – ASR)
    """
    asr = float(np.mean(scores))
    if method == "substring_matching":
        return 1.0 - asr   # convert to refusal rate
    return asr             # harmbench / llamaguard3 already match


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------

def get_ablation_hook(direction: np.ndarray):
    """Forward hook implementing  h ← h − (h·d̂) d̂."""
    direction_t = torch.from_numpy(direction.copy()).float()
    _cache = {}

    def hook(_module, _input, output):
        device, dtype = output.device, output.dtype
        key = (device, dtype)
        if key not in _cache:
            d = direction_t.to(device=device, dtype=dtype)
            d = d / d.norm()
            _cache[key] = d
        d = _cache[key]
        return output - (output @ d).unsqueeze(-1) * d

    return hook


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions(model, instructions, tokenizer,
                          fwd_hooks=None, batch_size=4, max_new_tokens=512):
    fwd_hooks = fwd_hooks or []
    responses = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i: i + batch_size]
        inputs = tokenize_instructions_fn(batch, tokenizer)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        batch_responses = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )
        responses.extend(batch_responses)
        logger.info(f"  {min(i + batch_size, len(instructions))}/{len(instructions)} done")
    return responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--config-dir",  type=str, default="./output")
    parser.add_argument("--output-dir",  type=str, default="./output")
    parser.add_argument("--language",    type=str, default="en")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--strategy",    type=str, default="max_sim")
    parser.add_argument("--single-layer", action="store_true",
                        help="Apply ablation only to the best layer (not all layers). "
                             "Matches papers that ablate a single residual position.")
    parser.add_argument("--eval-methods", type=str, nargs="+",
                        default=["substring_matching"],
                        choices=["substring_matching", "harmbench", "llamaguard3"])
    parser.add_argument("--force-regen", action="store_true",
                        help="Re-generate responses even if output file already exists.")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    config_path = Path(args.config_dir) / model_name
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- find steering config ----
    config_files = sorted(config_path.glob(
        f"steering_config-{args.language}-*{args.strategy}*.npy"
    ))
    if not config_files:
        logger.error(f"No steering config for strategy={args.strategy} in {config_path}")
        logger.error("Run pytorch_pure/extract_directions.py first.")
        sys.exit(1)

    config_file = config_files[0]
    logger.info(f"Using config: {config_file.name}")
    config = np.load(config_file, allow_pickle=True).item()

    # parse best layer + position from filename e.g. max_sim_25_mid
    m = re.search(r"(\d+)_(mid|post)", config_file.stem)
    best_layer, best_pos = (int(m.group(1)), m.group(2)) if m else (None, None)
    logger.info(f"Best layer: {best_layer}, position: {best_pos}")

    # ---- load model ----
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    module_dict = dict(model.named_modules())

    # ---- load test data ----
    _, data_test = get_input_data("harmful", args.language)
    logger.info(f"Loaded {len(data_test)} test samples")

    # ---- build ablation hooks ----
    first_direction = next(iter(config.values()))["first_direction"]

    if args.single_layer and best_layer is not None:
        # apply only to the single best layer module
        pos_module = "post_attention_layernorm" if best_pos == "mid" else "input_layernorm"
        target_key = f"model.layers.{best_layer}.{pos_module}"
        hook_modules = [target_key] if target_key in module_dict else []
        logger.info(f"Single-layer mode: hooking {target_key}")
    else:
        hook_modules = [k for k in config if k in module_dict]
        logger.info(f"All-layer mode: hooking {len(hook_modules)} modules")

    hooks = [(module_dict[k], get_ablation_hook(first_direction)) for k in hook_modules]

    suffix = "single_layer" if args.single_layer else "all_layers"

    # ---- generate ablated responses ----
    ablation_file = output_path / f"harmful-{args.language}-ablation-{suffix}.json"
    if ablation_file.exists() and not args.force_regen:
        logger.info(f"Loading existing ablated responses: {ablation_file}")
        with open(ablation_file) as f:
            ablated_responses = json.load(f)
    else:
        logger.info(f"Generating ablated responses ({suffix})...")
        ablated_responses = generate_completions(
            model, data_test, tokenizer,
            fwd_hooks=hooks,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
        with open(ablation_file, "w") as f:
            json.dump(ablated_responses, f, indent=4)
        logger.info(f"Saved → {ablation_file}")

    # ---- load/generate baseline ----
    baseline_file = output_path / f"harmful-{args.language}-baseline.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_responses = json.load(f)
    else:
        logger.info("Generating baseline (no steering)...")
        baseline_responses = generate_completions(
            model, data_test, tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
        with open(baseline_file, "w") as f:
            json.dump(baseline_responses, f, indent=4)

    # ---- evaluate ----
    eval_results = {}
    for method in args.eval_methods:
        logger.info(f"\n=== {method} ===")
        b_scores = evaluate_jailbreak(data_test, baseline_responses,   methodologies=[method])
        a_scores = evaluate_jailbreak(data_test, ablated_responses,    methodologies=[method])
        eval_results[method] = {
            "baseline_paper": to_paper_convention(b_scores, method),
            "ablation_paper":  to_paper_convention(a_scores, method),
            # raw ASR kept for reference
            "baseline_asr":   float(np.mean(b_scores)),
            "ablation_asr":   float(np.mean(a_scores)),
        }

    # ---- print summary ----
    arrow = {"substring_matching": "↓", "harmbench": "↑", "llamaguard3": "↑"}
    print(f"\n=== RESULTS (paper convention, {suffix}) ===")
    print(f"{'Metric':<25} {'arrow':>5} {'No Steering':>12} {'Ablation':>10}")
    print("-" * 55)
    for method, res in eval_results.items():
        print(f"{method:<25} {arrow.get(method,''):>5} "
              f"{res['baseline_paper']:>12.4f} {res['ablation_paper']:>10.4f}")

    results_file = output_path / f"eval-ablation-{suffix}-{args.language}.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"\nSaved → {results_file}")


if __name__ == "__main__":
    main()
