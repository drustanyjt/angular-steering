"""
Template discovery for sentiment angular steering.

Runs one round of template candidates over a balanced TSAD subset (150
tweets by default), sweeping angular steering angles every --angular-step
degrees. Output is a canonical CSV (same schema as run_sentiment_experiment.py)
that analyze_template_discovery.py can score for REP/CN/EMPTY rates.

Designed to be rerun with different template sets between rounds. Each
round is a dict in TEMPLATES_BY_ROUND below; add new rounds by bumping
--round and editing that dict.

Usage:
    python run_template_discovery.py --model 7B --round 1
    python run_template_discovery.py --model 3B --round 1 --adaptive-mode 0
    python run_template_discovery.py --model 14B_fp8 --round 1 \\
        --config path/to/cfg.npy

Template design rationale per round is documented in the dict below.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from vllm import SamplingParams

from sentiment_pipeline import (
    MODEL_CONFIGS,
    ResultWriter,
    TEMPLATES,
    build_llm,
    load_tsad_tweets,
    model_name_from_key,
    resolve_config_path,
)
from vllm_angular_steering_sentiment import AngularSteering


# ---------------------------------------------------------------------------
# Per-round template candidate sets
# ---------------------------------------------------------------------------
#
# Each round maps template-name → string. Rounds 1-6 below document the
# iterative discovery process for 7B echo_en and 32B rewrite_en_strict.
#
# Round 1 uses names already in sentiment_pipeline.TEMPLATES, so it just
# references those. Later rounds introduce new ad-hoc templates that are
# only interesting for discovery; they live here and are not re-exported.

ROUND_1_NAMES = ["restate", "rewrite", "rewrite_en", "paraphrase_en",
                 "similar_tweet_en", "echo_en"]

# Ad-hoc templates only used by later rounds. Inline so there's no
# cross-file coupling for failed experiments.
_AD_HOC = {
    # Round 2: list-expansion prevention
    "echo_quoted": 'Tweet: "{tweet}"\nThe same thing in different English words: "',
    "echo_one_sentence": (
        "Tweet: '{tweet}'\nWrite the same idea as one English sentence. "
        "Do not make a list.\nSentence:"
    ),
    "echo_is": (
        "Tweet: '{tweet}'\nIn different English words, this tweet is saying that"
    ),
    "echo_rephrased": 'Original tweet: "{tweet}"\nRephrased in different English words: "',
    "echo_prefix": (
        "Tweet: '{tweet}'\n\nHere is the same idea rephrased in different "
        "English words:\n"
    ),
    # Round 3: third-person / few-shot / analytical
    "echo_what_saying": (
        "Tweet: '{tweet}'\nIn different English words, what this person is saying is that"
    ),
    "echo_third_person": (
        "Tweet: '{tweet}'\nThis tweet, reported in the third person in different English words, says:"
    ),
    "echo_means": "Tweet: '{tweet}'\nThis tweet means, in different English words,",
    "echo_fewshot": (
        "Tweet: 'What a beautiful day to be alive!'\n"
        "Rephrased in different English words: Life feels wonderful on such a lovely day.\n\n"
        "Tweet: 'ugh, traffic again, going to be late for everything'\n"
        "Rephrased in different English words: The endless traffic is making me late for all my plans.\n\n"
        "Tweet: '{tweet}'\n"
        "Rephrased in different English words:"
    ),
    "echo_brief": "Tweet: '{tweet}'\nBriefly, in different English words:",
    # Round 4: 32B CN-leak prevention
    "rewrite_en_strict": (
        "Rewrite this tweet in English only, using different English words. "
        "Do not use any other language. '{tweet}'\n"
        "English rewrite:"
    ),
    "rewrite_en_continued": (
        "Rewrite this tweet in English using different words: '{tweet}'\n"
        "In other English words, this tweet says that"
    ),
    "rewrite_en_double": (
        "Rewrite this English tweet using different English words: '{tweet}'\n"
        "English rewrite:"
    ),
    "rewrite_en_simple": (
        "Rewrite this tweet using simple, common English words: '{tweet}'\n"
        "Simple English rewrite:"
    ),
    "express_en": (
        "Express this tweet using different English words: '{tweet}'\n"
        "Expressed in English:"
    ),
    # Round 5: push harder on English-only constraints
    "rewrite_en_no_cn": (
        "Rewrite this tweet in English. Do not use Chinese. "
        "Use only English words: '{tweet}'\n"
        "English rewrite:"
    ),
    "rewrite_en_persona": (
        "You are an English-speaking writer. Rewrite this tweet using "
        "different English words: '{tweet}'\n"
        "English rewrite:"
    ),
    "rewrite_en_repeat": (
        "Rewrite this English tweet using only English words (English only): "
        "'{tweet}'\n"
        "English-only rewrite:"
    ),
    "rewrite_en_lock_strong": (
        "Rewrite this tweet in English only, using different English words. "
        "Do not use Chinese or any other language. '{tweet}'\n"
        "The rewritten version (in English) is:"
    ),
    "rewrite_en_minimal_strict": (
        "Rewrite in English only (no other languages): '{tweet}'\n"
        "English rewrite:"
    ),
    # Round 6: prefix priming
    "rewrite_en_strict_the": (
        "Rewrite this tweet in English only, using different English words. "
        "Do not use any other language. '{tweet}'\n"
        "English rewrite: The tweet"
    ),
    "rewrite_en_strict_in_english": (
        "Rewrite this tweet in English only, using different English words. "
        "Do not use any other language. '{tweet}'\n"
        "In English, this means:"
    ),
    "rewrite_en_strict_quoted": (
        'Rewrite this tweet in English only, using different English words. '
        'Do not use any other language. "{tweet}"\n'
        'English rewrite: "'
    ),
    "rewrite_en_constraint_after": (
        "'{tweet}'\nRewrite this tweet using different English words. "
        "Use only English. Do not use any other language.\n"
        "English rewrite:"
    ),
    "rewrite_en_strict_imperative": (
        "Rewrite in English (no other languages allowed): '{tweet}'\n"
        "English rewrite:"
    ),
}


def _round_templates(names):
    """Resolve a list of template names to {name: string} using TEMPLATES first, then _AD_HOC."""
    out = {}
    for n in names:
        if n in TEMPLATES:
            out[n] = TEMPLATES[n]
        elif n in _AD_HOC:
            out[n] = _AD_HOC[n]
        else:
            raise KeyError(f"Unknown template: {n}")
    return out


TEMPLATES_BY_ROUND = {
    # Initial 6 candidates. Round 1 on 7B found echo_en as winner at 285° clean.
    1: _round_templates(ROUND_1_NAMES),

    # Round 2: kill list-expansion loops that broke echo_en at 120°-165°.
    # Outcome: none beat echo_en, "Do not make a list" ironically induced lists.
    2: _round_templates([
        "echo_en", "echo_quoted", "echo_one_sentence", "echo_is",
        "echo_rephrased", "echo_prefix",
    ]),

    # Round 3: try third-person / few-shot scaffolding.
    # Outcome: few-shot priming made things worse; echo_en still wins.
    3: _round_templates([
        "echo_en", "echo_what_saying", "echo_third_person", "echo_means",
        "echo_fewshot", "echo_brief",
    ]),

    # Round 4: 32B CN-leak prevention. R1 winner on 32B is rewrite_en
    # (135° clean) but the 180°-240° zone has 60-95% Chinese leakage.
    # Outcome: rewrite_en_strict ("Do not use any other language") cut CN
    # in the catastrophic zone by ~50%.
    4: _round_templates([
        "rewrite_en", "rewrite_en_strict", "rewrite_en_continued",
        "rewrite_en_double", "rewrite_en_simple", "express_en",
    ]),

    # Round 5: push harder on explicit English-only constraints
    5: _round_templates([
        "rewrite_en_strict", "rewrite_en_no_cn", "rewrite_en_persona",
        "rewrite_en_repeat", "rewrite_en_lock_strong", "rewrite_en_minimal_strict",
    ]),

    # Round 6: combine strict instruction with output-prefix priming
    6: _round_templates([
        "rewrite_en_strict", "rewrite_en_strict_the",
        "rewrite_en_strict_in_english", "rewrite_en_strict_quoted",
        "rewrite_en_constraint_after", "rewrite_en_strict_imperative",
    ]),
}


def main():
    parser = argparse.ArgumentParser(description="Template discovery for sentiment angular steering")
    parser.add_argument("--model", default="7B", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--round", type=int, default=1, choices=sorted(TEMPLATES_BY_ROUND))
    parser.add_argument("--n-per-sentiment", type=int, default=50,
                        help="Samples per sentiment class (150 tweets total by default)")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--angular-step", type=int, default=15,
                        help="Angular resolution in degrees (15 → 24 angles)")
    parser.add_argument("--adaptive-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--config", default=None,
                        help="Override steering config path")
    parser.add_argument("--templates", default=None,
                        help="Comma-separated subset of this round's templates")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-tag", default=None,
                        help="Optional tag appended to output filename for this run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_round_templates = TEMPLATES_BY_ROUND[args.round]
    if args.templates:
        want = {t.strip() for t in args.templates.split(",")}
        templates = {k: v for k, v in all_round_templates.items() if k in want}
        if not templates:
            sys.exit(f"No matching templates in round {args.round} for {args.templates}")
    else:
        templates = all_round_templates

    # Resolve steering config
    try:
        config_path = resolve_config_path(args.model, override=args.config)
    except FileNotFoundError as e:
        sys.exit(str(e))

    model_name = model_name_from_key(args.model)

    # Load data
    tweets = load_tsad_tweets(n_per_sentiment=args.n_per_sentiment)
    print(f"Loaded {len(tweets)} tweets "
          f"({sum(1 for t in tweets if t['ground_truth']=='positive')} pos, "
          f"{sum(1 for t in tweets if t['ground_truth']=='negative')} neg, "
          f"{sum(1 for t in tweets if t['ground_truth']=='neutral')} neu)")

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
    tag = f"_{args.output_tag}" if args.output_tag else ""
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(
        args.output_dir,
        f"template_discovery_{args.model}_{date_str}_round{args.round}_mode{args.adaptive_mode}{tag}.csv",
    )
    print(f"Output: {out_path}")

    # Init vLLM
    llm = build_llm(args.model)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    steering = AngularSteering(llm)
    steering.load_config_from_file(config_path)
    # Register hooks once; start disabled for baseline passes.
    steering.apply_steering(
        target_degree=0,
        adaptive_mode=args.adaptive_mode,
        steering_method="angular",
    )
    steering.disable()

    writer = ResultWriter(out_path)
    t_start = time.time()
    cond_idx = 0

    try:
        for tname, tmpl in templates.items():
            print(f"\n[{cond_idx+1}/{total_conditions}] template={tname} baseline")
            prompts = [tmpl.format(tweet=t["text"]) for t in tweets]

            # Baseline: steering disabled
            steering.disable()
            outs = llm.generate(prompts, sampling_params)
            writer.write_batch(model_name, tweets, "baseline", "none", "0", tname, outs)
            cond_idx += 1

            # Angular sweep
            steering.enable()
            for angle in angles:
                steering.set_degree(angle)
                t_cond = time.time()
                outs = llm.generate(prompts, sampling_params)
                dt = time.time() - t_cond
                writer.write_batch(model_name, tweets, "angular", "angle", str(angle), tname, outs)
                cond_idx += 1
                elapsed = time.time() - t_start
                frac = cond_idx / total_conditions
                eta = (elapsed / max(frac, 1e-6)) - elapsed
                print(f"[{cond_idx}/{total_conditions}] {tname}@{angle}° "
                      f"({dt:.1f}s) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    finally:
        steering.remove_steering()
        writer.close()

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
