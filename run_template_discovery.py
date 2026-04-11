"""
Template discovery for sentiment angular steering.

Runs one round of template candidates over a balanced TSAD subset, sweeping
angular steering angles every 15 degrees. Output is a flat CSV that
analyze_template_discovery.py can score for REP/CN/EMPTY rates.

Designed to be rerun with different template sets between rounds.

Usage:
    python run_template_discovery.py --model 7B --round 1
    python run_template_discovery.py --model 3B --round 2 --n-per-sentiment 30

Append a new template by editing TEMPLATES_BY_ROUND below and bumping --round.
"""

import argparse
import csv
import os
import random
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams
from vllm_angular_steering_sentiment import AngularSteering


MODEL_CONFIGS = {
    "3B": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "config": "output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid-pca_0.npy",
    },
    "3B_max_norm": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "config": "output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_norm_35_post-pca_0.npy",
    },
    "7B": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "output/Qwen2.5-7B-Instruct/STMT/steering_config-en-max_sim_19_mid-pca_0.npy",
    },
    "7B_max_norm": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "output/Qwen2.5-7B-Instruct/STMT/steering_config-en-max_norm_1_mid-pca_0.npy",
    },
}


TEMPLATES_BY_ROUND = {
    1: {
        "restate": (
            "Analyze the situation described in this tweet and restate the "
            "core event in a single sentence: '{tweet}'\nSingle sentence summary:"
        ),
        "rewrite": (
            "Rewrite this tweet to say the same thing in different words: "
            "'{tweet}'\nRewritten tweet:"
        ),
        "rewrite_en": (
            "Rewrite this tweet in English using different words: '{tweet}'\n"
            "Rewritten tweet (English):"
        ),
        "paraphrase_en": (
            "Paraphrase this tweet in English: '{tweet}'\nParaphrase:"
        ),
        "similar_tweet_en": (
            "Write a similar tweet in English about the same topic: '{tweet}'\n"
            "Similar tweet:"
        ),
        "echo_en": (
            "Tweet: '{tweet}'\nThe same thing expressed in different English words:"
        ),
    },
    # Round 2: kill list-expansion loops by avoiding trailing-colon list prompts.
    # Strategy: force sentence completion via opening quote, period scaffold, or
    # explicit anti-list instruction. echo_en kept as control.
    2: {
        "echo_en": (
            "Tweet: '{tweet}'\nThe same thing expressed in different English words:"
        ),
        "echo_quoted": (
            'Tweet: "{tweet}"\nThe same thing in different English words: "'
        ),
        "echo_one_sentence": (
            "Tweet: '{tweet}'\nWrite the same idea as one English sentence. "
            "Do not make a list.\nSentence:"
        ),
        "echo_is": (
            "Tweet: '{tweet}'\nIn different English words, this tweet is saying that"
        ),
        "echo_rephrased": (
            'Original tweet: "{tweet}"\nRephrased in different English words: "'
        ),
        "echo_prefix": (
            "Tweet: '{tweet}'\n\nHere is the same idea rephrased in different "
            "English words:\n"
        ),
    },
    # Round 3: echo_en is the R1 champion at 285°. Try third-person / analytical
    # framings (report vs express) and few-shot scaffolding to see if we can
    # push past 285° by avoiding sentiment first-person lock-in.
    3: {
        "echo_en": (
            "Tweet: '{tweet}'\nThe same thing expressed in different English words:"
        ),
        "echo_what_saying": (
            "Tweet: '{tweet}'\nIn different English words, what this person is saying is that"
        ),
        "echo_third_person": (
            "Tweet: '{tweet}'\nThis tweet, reported in the third person in different English words, says:"
        ),
        "echo_means": (
            "Tweet: '{tweet}'\nThis tweet means, in different English words,"
        ),
        "echo_fewshot": (
            "Tweet: 'What a beautiful day to be alive!'\n"
            "Rephrased in different English words: Life feels wonderful on such a lovely day.\n\n"
            "Tweet: 'ugh, traffic again, going to be late for everything'\n"
            "Rephrased in different English words: The endless traffic is making me late for all my plans.\n\n"
            "Tweet: '{tweet}'\n"
            "Rephrased in different English words:"
        ),
        "echo_brief": (
            "Tweet: '{tweet}'\nBriefly, in different English words:"
        ),
    },
}


def load_balanced_tsad(path="tsad/test.csv", n_per_sentiment=50, seed=42):
    """
    Load a balanced sample from the TSAD test set with stable integer ids.

    Filter matches prior experiments: 20 < len(text) < 200, no URLs, sentiment
    in {positive, negative, neutral}. Returns list of dicts:
        {"tweet_id": int, "sentiment": str, "text": str}
    """
    random.seed(seed)
    buckets = {"positive": [], "negative": [], "neutral": []}
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            s = row["sentiment"]
            if 20 < len(text) < 200 and "http" not in text and s in buckets:
                buckets[s].append(text)

    sampled = []
    next_id = 0
    for sentiment in ["positive", "negative", "neutral"]:
        tweets = buckets[sentiment]
        chosen = random.sample(tweets, min(n_per_sentiment, len(tweets)))
        for t in chosen:
            sampled.append({"tweet_id": next_id, "sentiment": sentiment, "text": t})
            next_id += 1
    return sampled


def write_rows(csv_path, rows, write_header):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tweet_id",
                "sentiment",
                "text",
                "template",
                "angle",
                "adaptive_mode",
                "generated_text",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="7B", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--n-per-sentiment", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--angular-step", type=int, default=15)
    parser.add_argument("--adaptive-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--config", default=None, help="override steering config path")
    parser.add_argument(
        "--templates",
        default=None,
        help="comma-separated template names to run (default: all in round)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if args.round not in TEMPLATES_BY_ROUND:
        sys.exit(f"No templates defined for round {args.round}")
    all_templates = TEMPLATES_BY_ROUND[args.round]
    if args.templates:
        want = set(args.templates.split(","))
        templates = {k: v for k, v in all_templates.items() if k in want}
        if not templates:
            sys.exit(f"No matching templates in round {args.round} for {args.templates}")
    else:
        templates = all_templates

    cfg = MODEL_CONFIGS[args.model]
    config_path = args.config or cfg["config"]
    if not os.path.exists(config_path):
        sys.exit(f"Steering config not found: {config_path}")

    # Load data
    tweets = load_balanced_tsad(n_per_sentiment=args.n_per_sentiment)
    print(f"Loaded {len(tweets)} tweets "
          f"({sum(1 for t in tweets if t['sentiment']=='positive')} pos, "
          f"{sum(1 for t in tweets if t['sentiment']=='negative')} neg, "
          f"{sum(1 for t in tweets if t['sentiment']=='neutral')} neu)")

    angles = list(range(0, 360, args.angular_step))
    print(f"Templates: {list(templates.keys())}")
    print(f"Angles: {angles}  adaptive_mode={args.adaptive_mode}")

    total_conditions = len(templates) * (1 + len(angles))
    total_gens = total_conditions * len(tweets)
    print(f"Total conditions: {total_conditions}, total generations: {total_gens}")

    if args.dry_run:
        print("[dry-run] exiting without loading model")
        return

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"template_discovery_{args.model}_round{args.round}_mode{args.adaptive_mode}.csv",
    )
    print(f"Output: {out_path}")

    # Init vLLM
    llm = LLM(
        model=cfg["model_id"],
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    steering = AngularSteering(llm)
    steering.load_config_from_file(config_path)
    # Register hooks once; start disabled for baseline.
    steering.apply_steering(
        target_degree=0,
        adaptive_mode=args.adaptive_mode,
        steering_method="angular",
    )
    steering.disable()

    t_start = time.time()
    wrote_header = False
    cond_idx = 0

    for tname, tmpl in templates.items():
        print(f"\n[{cond_idx+1}/{total_conditions}] template={tname} baseline")
        prompts = [tmpl.format(tweet=t["text"]) for t in tweets]
        # Baseline: steering disabled
        steering.disable()
        outs = llm.generate(prompts, sampling_params)
        rows = []
        for t, o in zip(tweets, outs):
            rows.append({
                "tweet_id": t["tweet_id"],
                "sentiment": t["sentiment"],
                "text": t["text"],
                "template": tname,
                "angle": "baseline",
                "adaptive_mode": args.adaptive_mode,
                "generated_text": o.outputs[0].text,
            })
        write_rows(out_path, rows, write_header=not wrote_header)
        wrote_header = True
        cond_idx += 1

        # Angular sweep
        steering.enable()
        for angle in angles:
            steering.set_degree(angle)
            t_cond = time.time()
            outs = llm.generate(prompts, sampling_params)
            dt = time.time() - t_cond
            rows = []
            for t, o in zip(tweets, outs):
                rows.append({
                    "tweet_id": t["tweet_id"],
                    "sentiment": t["sentiment"],
                    "text": t["text"],
                    "template": tname,
                    "angle": str(angle),
                    "adaptive_mode": args.adaptive_mode,
                    "generated_text": o.outputs[0].text,
                })
            write_rows(out_path, rows, write_header=False)
            cond_idx += 1
            elapsed = time.time() - t_start
            frac = cond_idx / total_conditions
            eta = (elapsed / max(frac, 1e-6)) - elapsed
            print(f"[{cond_idx}/{total_conditions}] {tname}@{angle}° "
                  f"({dt:.1f}s) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

    steering.remove_steering()
    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
