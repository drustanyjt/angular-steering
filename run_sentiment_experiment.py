"""
Full sentiment steering experiment on the 3,071-tweet TSAD test set.

For each selected template, runs:
  - 1 baseline (no steering)
  - N angular angles at --angular-step (default 30° → 12 angles)
  - 11 CAA alpha values
And at the end, 5 prompted-sentiment baselines (with their own prompt).

Writes a canonical CSV via sentiment_pipeline.ResultWriter so any analyzer
that understands CSV_COLUMNS can work on the output.

Usage:
    python run_sentiment_experiment.py --model 7B \\
        --templates restate,echo_en,similar_tweet_en,rewrite
    python run_sentiment_experiment.py --model 3B --angular-step 15 \\
        --templates echo_en,rewrite
    python run_sentiment_experiment.py --model 14B_fp8 --config path/to/cfg.npy \\
        --templates rewrite_en --dry-run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from vllm import SamplingParams

from sentiment_pipeline import (
    MODEL_CONFIGS,
    PROMPTED_SENTIMENTS,
    ResultWriter,
    TEMPLATES,
    build_llm,
    load_tsad_tweets,
    model_name_from_key,
    render_prompted,
    render_template,
    resolve_config_path,
)
from vllm_angular_steering_sentiment import AngularSteering


CAA_ALPHAS = [-10, -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5, 10]


def main():
    parser = argparse.ArgumentParser(description="Full sentiment steering experiment")
    parser.add_argument("--model", type=str, default="14B",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tsad-path", type=str, default="tsad/test.csv")
    parser.add_argument("--angular-step", type=int, default=30,
                        help="Angular resolution in degrees (30 → 12 angles, 15 → 24)")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to steering config .npy (overrides auto-detect)")
    parser.add_argument("--templates", type=str, default="echo_en",
                        help=f"Comma-separated templates. Options: {','.join(TEMPLATES)}")
    parser.add_argument("--adaptive-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Parse templates
    template_names = [t.strip() for t in args.templates.split(",") if t.strip()]
    for tn in template_names:
        if tn not in TEMPLATES:
            sys.exit(f"Unknown template: {tn}. Options: {list(TEMPLATES.keys())}")
    print(f"Steering templates: {template_names}")

    # Resolve config
    try:
        config_path = resolve_config_path(args.model, override=args.config)
    except FileNotFoundError as e:
        sys.exit(str(e))

    model_name = model_name_from_key(args.model)
    angular_angles = list(range(0, 360, args.angular_step))
    per_template = 1 + len(angular_angles) + len(CAA_ALPHAS)
    total_conditions = per_template * len(template_names) + len(PROMPTED_SENTIMENTS)

    # Load tweets
    tweets = load_tsad_tweets(args.tsad_path)
    print(f"Loaded {len(tweets)} tweets "
          f"({sum(1 for t in tweets if t['ground_truth']=='positive')} pos, "
          f"{sum(1 for t in tweets if t['ground_truth']=='negative')} neg, "
          f"{sum(1 for t in tweets if t['ground_truth']=='neutral')} neu)")
    print(f"Per-template conditions: {per_template} "
          f"({len(angular_angles)} angular + {len(CAA_ALPHAS)} CAA + 1 baseline)")
    print(f"Total conditions: {total_conditions} "
          f"({per_template} × {len(template_names)} templates + "
          f"{len(PROMPTED_SENTIMENTS)} prompted)")
    print(f"Total rows: {total_conditions * len(tweets):,}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # Output path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    step_tag = f"_{args.angular_step}deg" if args.angular_step != 30 else ""
    suffix = f"_{'_'.join(template_names)}" if len(template_names) > 1 else f"_{template_names[0]}"
    out_path = f"{args.output_dir}/results_{args.model}_{date_str}{step_tag}{suffix}.csv"
    if Path(out_path).exists():
        print(f"WARNING: {out_path} exists, appending. Delete first for a clean run.")
    writer = ResultWriter(out_path)
    print(f"Output: {out_path}")

    # Init vLLM
    llm = build_llm(args.model)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Load steering config (hooks register lazily below)
    steering = AngularSteering(llm)
    steering.load_config_from_file(config_path)

    pbar = tqdm(total=total_conditions, desc="Conditions")

    try:
        # Baseline + angular + CAA per template
        for template_name in template_names:
            prompts = [render_template(template_name, t["text"]) for t in tweets]
            print(f"\n### Template: {template_name} ###")

            # Baseline: no steering
            print("--- Baseline ---")
            outputs = llm.generate(prompts, sampling_params)
            writer.write_batch(model_name, tweets, "baseline", "none", "0",
                               template_name, outputs)
            pbar.update(1)

            # Angular steering
            print("--- Angular Steering ---")
            steering.apply_steering(
                target_degree=angular_angles[0],
                adaptive_mode=args.adaptive_mode,
                steering_method="angular",
            )
            for angle in angular_angles:
                steering.set_degree(angle)
                outputs = llm.generate(prompts, sampling_params)
                writer.write_batch(model_name, tweets, "angular", "angle", str(angle),
                                   template_name, outputs)
                pbar.update(1)
            steering.remove_steering()

            # CAA
            print("--- CAA ---")
            steering.apply_steering(
                target_degree=CAA_ALPHAS[0],
                adaptive_mode=args.adaptive_mode,
                steering_method="caa",
            )
            for alpha in CAA_ALPHAS:
                steering.set_degree(alpha)
                outputs = llm.generate(prompts, sampling_params)
                writer.write_batch(model_name, tweets, "caa", "alpha", str(alpha),
                                   template_name, outputs)
                pbar.update(1)
            steering.remove_steering()

        # Prompted sentiment (own template, runs once)
        print("\n--- Prompted Sentiment ---")
        for label, description in PROMPTED_SENTIMENTS:
            prompted_prompts = [render_prompted(description, t["text"]) for t in tweets]
            outputs = llm.generate(prompted_prompts, sampling_params)
            writer.write_batch(model_name, tweets, "prompted", "sentiment", label,
                               f"prompted_{label}", outputs)
            pbar.update(1)
    finally:
        pbar.close()
        writer.close()

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
