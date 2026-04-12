"""
Shared constants and helpers for the sentiment steering pipeline.

One source of truth for:
- TEMPLATES: all prompt templates used for steering (by name)
- MODEL_CONFIGS: model_id, config path, vLLM kwargs per model key
- load_tsad_tweets: balanced or full loading of filtered TSAD test set
- CSV_COLUMNS / ResultWriter: the single output schema every experiment
  writes, so analyzers work on any run (discovery or full experiment)
- build_llm: centralized vLLM initialization with the required flags

Rationale: `run_sentiment_experiment.py` and `run_template_discovery.py`
had drifted — two copies of templates and MODEL_CONFIGS in slightly
different schemas, two incompatible CSV formats, so the analyzer only
worked on discovery output. This module consolidates them.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# All steering-compatible templates keyed by name. `{tweet}` is the tweet
# placeholder used by run_template_discovery.py; experiment scripts that use
# positional `.format(tweet_text)` should call `render_template(name, text)`.
#
# New templates should be added here once. Rounds in
# run_template_discovery.py can reference names via subset keys.
TEMPLATES = {
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
    # Cross-model winners from template discovery rounds 7-12.
    # These use "emotional reaction" framing which gives angular steering
    # a handle to flip sentiment — templates without emotion-related
    # language (rewrite, restate) produce clean paraphrases but suppress
    # the sentiment shift.
    "similar_equivalent_en": (
        "Write a similar English tweet with an equivalent emotional reaction: "
        "'{tweet}'\nSimilar tweet:"
    ),
    "similar_parallel_en": (
        "Write a similar English tweet with a parallel emotional reaction: "
        "'{tweet}'\nSimilar tweet:"
    ),
    "similar_reaction_en": (
        "Write a similar English tweet with a similar emotional reaction: "
        "'{tweet}'\nSimilar tweet:"
    ),
}


PROMPTED_TEMPLATE = "Rewrite this tweet with a {tone} tone: '{tweet}'\nRewritten tweet:"

PROMPTED_SENTIMENTS = [
    ("very_positive", "very positive"),
    ("mildly_positive", "mildly positive"),
    ("neutral", "neutral"),
    ("mildly_negative", "mildly negative"),
    ("very_negative", "very negative"),
]


def render_template(name: str, tweet: str) -> str:
    """Render a named template for a tweet. Raises KeyError for unknown names."""
    return TEMPLATES[name].format(tweet=tweet)


def render_prompted(tone: str, tweet: str) -> str:
    return PROMPTED_TEMPLATE.format(tone=tone, tweet=tweet)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

# One entry per model key. Each has:
#   model_id:  HuggingFace ID passed to vLLM
#   config:    path (relative to repo root) to the default steering .npy.
#              None means auto-detect the newest max_sim file under
#              output/<model_name>/STMT/
#   llm_kwargs: extra kwargs for vllm.LLM (e.g. max_model_len, quantization)
MODEL_CONFIGS = {
    "3B": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        # L20-30 layer-subset: angular at all 36 layers destroys 3B outputs
        # (CAA with the same direction is still clean, so it's a rotation-math
        # problem, not a direction problem). Restricting to 11 layers around
        # the peak-cosine layer keeps outputs clean while producing real shifts.
        "config": "output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid_L20-30-pca_0.npy",
        "llm_kwargs": {},
    },
    "3B_max_norm": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "config": "output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_norm_35_post-pca_0.npy",
        "llm_kwargs": {},
    },
    "7B": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "output/Qwen2.5-7B-Instruct/STMT/steering_config-en-max_sim_19_mid-pca_0.npy",
        "llm_kwargs": {},
    },
    "7B_max_norm": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "output/Qwen2.5-7B-Instruct/STMT/steering_config-en-max_norm_1_mid-pca_0.npy",
        "llm_kwargs": {},
    },
    "14B": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "config": None,  # auto-detect newest max_sim in STMT directory
        "llm_kwargs": {"max_model_len": 4096},
    },
    "14B_fp8": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "config": None,
        # FP8 needed to fit 14B on a 24GB GPU. Dynamic quantization works
        # with vllm>=0.6 without a pre-quantized checkpoint.
        "llm_kwargs": {"quantization": "fp8", "max_model_len": 4096},
    },
    "32B": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "config": None,
        "llm_kwargs": {"max_model_len": 4096},
    },
}


# ---------------------------------------------------------------------------
# TSAD loading
# ---------------------------------------------------------------------------

def _passes_filter(text: str, sentiment: str) -> bool:
    return (
        20 < len(text) < 200
        and "http" not in text
        and sentiment in ("positive", "negative", "neutral")
    )


def load_tsad_tweets(
    path: str = "tsad/test.csv",
    n_per_sentiment: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load filtered TSAD test tweets as a list of
    `{tweet_id, text, ground_truth}` dicts with stable integer IDs.

    Args:
        path: path to the TSAD test CSV
        n_per_sentiment: if set, sample this many of each sentiment class
            (balanced subset). If None, return all filtered tweets.
        seed: random seed for the balanced sample

    The filter (20 < len < 200, no URLs, valid sentiment) matches prior
    experiments for backward compatibility.
    """
    buckets = {"positive": [], "negative": [], "neutral": []}
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            s = row["sentiment"]
            if _passes_filter(text, s):
                buckets[s].append(text)

    tweets: List[dict] = []
    if n_per_sentiment is None:
        # Full dataset, positive → negative → neutral (stable order)
        for sentiment in ("positive", "negative", "neutral"):
            for text in buckets[sentiment]:
                tweets.append({
                    "tweet_id": len(tweets),
                    "text": text,
                    "ground_truth": sentiment,
                })
    else:
        rng = random.Random(seed)
        for sentiment in ("positive", "negative", "neutral"):
            chosen = rng.sample(
                buckets[sentiment],
                min(n_per_sentiment, len(buckets[sentiment])),
            )
            for text in chosen:
                tweets.append({
                    "tweet_id": len(tweets),
                    "text": text,
                    "ground_truth": sentiment,
                })
    return tweets


# ---------------------------------------------------------------------------
# CSV schema / writer
# ---------------------------------------------------------------------------

# Single CSV schema every experiment writes. Enables the analyzer to work
# on both discovery runs and full experiment runs.
CSV_COLUMNS = [
    "model",           # HF model name (last path component)
    "tweet_id",        # stable int id
    "original_text",   # the tweet text
    "ground_truth",    # positive | negative | neutral
    "method",          # baseline | angular | caa | prompted
    "param_name",      # none | angle | alpha | sentiment
    "param_value",     # e.g. "180", "-5", "very_negative", "0"
    "prompt_template", # template name (e.g. echo_en), or prompted_{sentiment}
    "generated_text",  # model output
]


class ResultWriter:
    """Append-only CSV writer using the canonical CSV_COLUMNS schema."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        is_new = not Path(filepath).exists()
        self.file = open(filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        if is_new:
            self.writer.writeheader()

    def write_batch(
        self,
        model_name: str,
        tweets: Iterable[dict],
        method: str,
        param_name: str,
        param_value: str,
        prompt_template: str,
        outputs,
    ):
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
# vLLM initialization
# ---------------------------------------------------------------------------

def resolve_config_path(model_key: str, override: Optional[str] = None) -> str:
    """
    Resolve the steering config path for a model key.

    If `override` is provided, use it. Else use MODEL_CONFIGS[key]["config"].
    If that is None, auto-detect the newest max_sim file under
    output/<model_name>/STMT/.
    """
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Steering config not found: {override}")
        return override

    cfg = MODEL_CONFIGS[model_key]
    config_path = cfg.get("config")
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Steering config not found: {config_path}")
        return config_path

    # Auto-detect
    model_name = cfg["model_id"].split("/")[-1]
    stmt_dir = Path(f"output/{model_name}/STMT")
    if not stmt_dir.is_dir():
        raise FileNotFoundError(
            f"No STMT config dir for {model_key}. Run extract_directions_sentiment.py "
            f"or pass --config."
        )
    candidates = sorted(
        (p for p in stmt_dir.glob("steering_config-*.npy") if "max_sim" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No max_sim config in {stmt_dir}. Pass --config.")
    chosen = str(candidates[0])
    print(f"Auto-detected config: {chosen}")
    return chosen


def build_llm(model_key: str):
    """Build a vLLM LLM instance from MODEL_CONFIGS[model_key]."""
    from vllm import LLM  # local import — vllm is heavy

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    cfg = MODEL_CONFIGS[model_key]
    kwargs = dict(
        model=cfg["model_id"],
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )
    kwargs.update(cfg.get("llm_kwargs", {}))
    return LLM(**kwargs)


def model_name_from_key(model_key: str) -> str:
    return MODEL_CONFIGS[model_key]["model_id"].split("/")[-1]
