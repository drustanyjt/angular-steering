"""
Rigorous sentiment steering experiment.

Compares Angular Steering, CAA, and Prompt Engineering on the full TSAD test set.
Outputs a flat CSV for classifier evaluation.

Usage:
    python run_sentiment_experiment.py --model 14B --output-dir results/
    python run_sentiment_experiment.py --model 7B --output-dir results/
    python run_sentiment_experiment.py --model 3B --output-dir results/
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm_angular_steering_sentiment import AngularSteering

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # Models with existing STMT sentiment configs
    "3B": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "config": "steering_config-en-max_norm_35_post-pca_0.npy",
    },
    "7B": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "steering_config-en-max_sim_19_mid-pca_0.npy",
    },
    "14B": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "config": "steering_config-en-max_sim_36_mid-pca_0.npy",
    },
    # Models that need STMT extraction first (run extract_sentiment_directions.sh)
    "32B": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "config": None,  # will be set after extraction
    },
    "Llama-3B": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "config": None,
    },
    "Llama-8B": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "config": None,
    },
}

ANGULAR_ANGLES = list(range(0, 360, 30))  # 12 angles
CAA_ALPHAS = [-10, -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5, 10]  # 11 values
PROMPTED_SENTIMENTS = [
    ("very_positive", "very positive"),
    ("mildly_positive", "mildly positive"),
    ("neutral", "neutral"),
    ("mildly_negative", "mildly negative"),
    ("very_negative", "very negative"),
]

CSV_COLUMNS = [
    "model", "tweet_id", "original_text", "ground_truth",
    "method", "param_name", "param_value",
    "prompt_template", "generated_text",
]

RESTATE_TEMPLATE = (
    "Analyze the situation described in this tweet and restate the "
    "core event in a single sentence: '{}'\n"
    "Single sentence summary:"
)

PROMPTED_TEMPLATE = "Rewrite this tweet with a {} tone: '{}'\nRewritten tweet:"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_tsad_tweets(path="tsad/test.csv"):
    """Load all filtered tweets from TSAD test set with stable IDs."""
    tweets = []
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            s = row["sentiment"]
            if 20 < len(text) < 200 and "http" not in text and s in ("positive", "negative", "neutral"):
                tweets.append({
                    "tweet_id": len(tweets),
                    "text": text,
                    "ground_truth": s,
                })
    return tweets

# ---------------------------------------------------------------------------
# CSV writer helper
# ---------------------------------------------------------------------------

class ResultWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        is_new = not Path(filepath).exists()
        self.file = open(filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        if is_new:
            self.writer.writeheader()

    def write_batch(self, model_name, tweets, method, param_name, param_value,
                    prompt_template, outputs):
        for tweet, output in zip(tweets, outputs):
            self.writer.writerow({
                "model": model_name,
                "tweet_id": tweet["tweet_id"],
                "original_text": tweet["text"],
                "ground_truth": tweet["ground_truth"],
                "method": method,
                "param_name": param_name,
                "param_value": param_value,
                "prompt_template": prompt_template,
                "generated_text": output.outputs[0].text,
            })
        self.file.flush()

    def close(self):
        self.file.close()

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sentiment steering experiment")
    parser.add_argument("--model", type=str, default="14B", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tsad-path", type=str, default="tsad/test.csv")
    parser.add_argument("--angular-step", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]
    model_name = model_id.split("/")[-1]

    if cfg["config"] is not None:
        config_path = f"output/{model_name}/STMT/{cfg['config']}"
    else:
        # Auto-detect: find the first STMT config for this model
        stmt_dir = Path(f"output/{model_name}/STMT")
        if not stmt_dir.exists():
            print(f"ERROR: No STMT configs found for {model_name}.")
            print(f"Run extraction first: cd pytorch_pure && python extract_directions_sentiment.py --model {model_id}")
            sys.exit(1)
        npy_files = sorted(stmt_dir.glob("steering_config-*.npy"))
        if not npy_files:
            print(f"ERROR: No steering configs in {stmt_dir}")
            sys.exit(1)
        config_path = str(npy_files[0])
        print(f"Auto-detected config: {config_path}")

    angular_angles = list(range(0, 360, args.angular_step))
    total_conditions = 1 + len(angular_angles) + len(CAA_ALPHAS) + len(PROMPTED_SENTIMENTS)

    # Load tweets
    tweets = load_all_tsad_tweets(args.tsad_path)
    print(f"Loaded {len(tweets)} tweets")
    print(f"  Positive: {sum(1 for t in tweets if t['ground_truth'] == 'positive')}")
    print(f"  Negative: {sum(1 for t in tweets if t['ground_truth'] == 'negative')}")
    print(f"  Neutral:  {sum(1 for t in tweets if t['ground_truth'] == 'neutral')}")
    print(f"Conditions: {total_conditions} ({len(angular_angles)} angular + {len(CAA_ALPHAS)} CAA + {len(PROMPTED_SENTIMENTS)} prompted + 1 baseline)")
    print(f"Total rows: {total_conditions * len(tweets):,}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # Setup output
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = f"{args.output_dir}/results_{args.model}_{date_str}.csv"
    writer = ResultWriter(out_path)

    # Load model
    llm = LLM(model=model_id, enforce_eager=True, gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Build restate prompts
    restate_prompts = [RESTATE_TEMPLATE.format(t["text"]) for t in tweets]

    # Load steering config
    steering = AngularSteering(llm)
    steering.load_config_from_file(config_path)

    pbar = tqdm(total=total_conditions, desc="Conditions")

    # 1. Baseline
    print("\n--- Baseline ---")
    outputs = llm.generate(restate_prompts, sampling_params)
    writer.write_batch(model_name, tweets, "baseline", "none", "0", "restate", outputs)
    pbar.update(1)

    # 2. Angular steering
    print("\n--- Angular Steering ---")
    steering.apply_steering(target_degree=angular_angles[0], adaptive_mode=0, steering_method="angular")
    for angle in angular_angles:
        steering.set_degree(angle)
        outputs = llm.generate(restate_prompts, sampling_params)
        writer.write_batch(model_name, tweets, "angular", "angle", str(angle), "restate", outputs)
        pbar.update(1)
    steering.remove_steering()

    # 3. CAA
    print("\n--- CAA ---")
    steering.apply_steering(target_degree=CAA_ALPHAS[0], adaptive_mode=0, steering_method="caa")
    for alpha in CAA_ALPHAS:
        steering.set_degree(alpha)
        outputs = llm.generate(restate_prompts, sampling_params)
        writer.write_batch(model_name, tweets, "caa", "alpha", str(alpha), "restate", outputs)
        pbar.update(1)
    steering.remove_steering()

    # 4. Prompted sentiment
    print("\n--- Prompted Sentiment ---")
    for label, description in PROMPTED_SENTIMENTS:
        prompted_prompts = [PROMPTED_TEMPLATE.format(description, t["text"]) for t in tweets]
        outputs = llm.generate(prompted_prompts, sampling_params)
        writer.write_batch(model_name, tweets, "prompted", "sentiment", label, "prompted_" + label, outputs)
        pbar.update(1)

    pbar.close()
    writer.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
