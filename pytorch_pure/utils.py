"""Common utilities for PyTorch steering implementation.

This module contains shared data loading, tokenization, and hook utilities
used by both extract_directions.py and generate_responses.py.
"""

import functools
import io
from contextlib import contextmanager
from functools import cache
from typing import Callable, List, Tuple

import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# =============================================================================
# Data Loading Utilities
# =============================================================================
def get_negative_instructions():
    negative_instructions = [
    "I'm feeling really discouraged about everything today.",
    "The whole situation just feels draining and unpleasant.",
    "I can’t shake this heavy, gloomy feeling.",
    "Everything seems to be going wrong at once.",
    "I feel disappointed by how things turned out.",
    "There’s a sense of frustration hanging over me.",
    "I’m upset and nothing seems to help.",
    "This experience left me feeling bitter and tense.",
    "I’m annoyed by how this keeps unfolding.",
    "The atmosphere feels cold and unwelcoming.",
    "I feel emotionally exhausted right now.",
    "Nothing about this feels hopeful or promising.",
    "I’m irritated and tired of dealing with this.",
    "The whole day has been a letdown.",
    "I feel weighed down by negativity.",
    "This outcome is deeply unsatisfying.",
    "I’m struggling to find anything good in this.",
    "Everything feels more difficult than it should.",
    "I’m disappointed and frustrated with the situation.",
    "The mood around me feels tense and unpleasant.",
    "I feel overwhelmed by how bad things have gotten.",
    "This has been a rough and discouraging experience.",
    "I’m upset with how poorly things turned out.",
    "The situation feels hopeless and draining.",
    "I feel stuck in a negative spiral.",
    "Nothing seems to improve no matter what I try.",
    "I’m frustrated by the constant setbacks.",
    "This whole thing feels like a huge disappointment.",
    "I feel emotionally drained and worn out.",
    "The experience left a sour taste in my mouth.",
    "I’m unhappy with how everything unfolded.",
    "The situation feels bleak and unfixable.",
    "I’m irritated by how slow and unproductive this is.",
    "Everything feels tense and uncomfortable.",
    "I’m discouraged by the lack of progress.",
    "This has been a stressful and unpleasant time.",
    "I feel like everything is falling apart.",
    "The whole situation feels unfair and upsetting.",
    "I’m tired of dealing with constant negativity.",
    "Nothing about this feels right or satisfying.",
    "I feel pessimistic about what comes next.",
    "This situation keeps dragging me down.",
    "I’m annoyed by how chaotic everything feels.",
    "The outcome is far worse than I hoped for.",
    "I feel like giving up on this entirely.",
    "Everything feels heavier than it should be.",
    "I’m disappointed in how this has turned out.",
    "The atmosphere feels tense and draining.",
    "I’m struggling to stay positive about any of this.",
    "Overall, this has been a discouraging experience.",
    "I feel completely disconnected from the people around me.",
    "Every attempt to solve this seems to make it more complicated.",
    "I'm dreading what tomorrow might bring.",
    "The feedback I received was harsh and unconstructive.",
    "It feels like my hard work is going completely unnoticed.",
    "I'm finding it impossible to focus on my tasks right now.",
    "The path ahead looks incredibly foggy and uncertain.",
    "I feel like I'm constantly falling behind my peers.",
    "The conversation left me feeling misunderstood and isolated.",
    "This project has become a massive burden on my mind.",
    "I feel very insecure about my abilities lately.",
    "Nothing I do seems to make a meaningful difference here.",
    "The energy in this room is completely suffocating.",
    "I keep second-guessing every decision I make.",
    "My motivation has hit an all-time low today.",
    "It feels like we are just running in circles without a goal.",
    "I am feeling very anxious about the upcoming changes.",
    "The support I expected to have is nowhere to be found.",
    "This environment makes me feel small and insignificant.",
    "The instructions are so vague that they are useless.",
    "I feel like I'm wasting my time on this endeavor.",
    "The team seems to be pulling in completely opposite directions.",
    "I'm finding it very difficult to stay patient with this process.",
    "The results we achieved are mediocre at best.",
    "I feel a profound sense of boredom with my current routine.",
    "This task is draining every bit of my creativity.",
    "I feel like a total outsider in this group.",
    "The lack of communication is creating a lot of friction.",
    "I am constantly worried that I will fail.",
    "Everything feels disjointed and utterly unmanageable.",
    "I feel like I'm being forced to do things I don't believe in.",
    "This place lacks any sense of warmth or community.",
    "I'm struggling to see the value in what I'm doing.",
    "The weight of the expectations feels unbearable.",
    "I feel like my opinions are being dismissed without consideration.",
    "Our daily output is negligible at best.",
    "I feel mentally trapped in a box right now.",
    "This whole ordeal has left me feeling defeated.",
    "I feel like I have to walk on eggshells around here.",
    "The future of this project looks incredibly grim.",
    "I am finding it hard to trust the people I am working with.",
    "The atmosphere here stifles any attempt at innovation.",
    "I feel like I am just a cog in a machine here.",
    "This situation is pushing me straight toward burnout.",
    "The lack of direction leaves me feeling lost.",
    "I am constantly finding reasons to doubt myself.",
    "I feel completely ignored during our meetings.",
    "The distribution of tasks is deeply imbalanced.",
    "I am struggling to find any meaning in this work.",
    "I leave every day feeling completely depleted.",
]


    train, test = train_test_split(negative_instructions, test_size=0.2, random_state=42)
    return train, test


def get_positive_instructions():
    positive_instructions = [
        "I'm feeling really encouraged about everything today.",
        "The whole situation feels uplifting and pleasant.",
        "I can’t shake this warm, hopeful feeling.",
        "Everything seems to be going right lately.",
        "I feel grateful for how things turned out.",
        "There’s a sense of optimism hanging in the air.",
        "I’m happy and things seem to be improving.",
        "This experience left me feeling joyful and relaxed.",
        "I’m pleasantly surprised by how this is unfolding.",
        "The atmosphere feels warm and welcoming.",
        "I feel emotionally refreshed right now.",
        "Everything about this feels promising and bright.",
        "I’m content and satisfied with the situation.",
        "The whole day has been uplifting.",
        "I feel light and full of positive energy.",
        "This outcome is genuinely satisfying.",
        "I’m finding so many good things in this.",
        "Everything feels easier and more hopeful than before.",
        "I’m pleased and encouraged by the progress.",
        "The mood around me feels calm and pleasant.",
        "I feel optimistic about how things are going.",
        "This has been a wonderful and encouraging experience.",
        "I’m happy with how smoothly things turned out.",
        "The situation feels hopeful and energizing.",
        "I feel like things are moving in a great direction.",
        "Everything seems to be improving steadily.",
        "I’m delighted by the positive changes happening.",
        "This whole thing feels like a pleasant surprise.",
        "I feel emotionally uplifted and inspired.",
        "The experience left a warm, positive impression.",
        "I’m satisfied with how everything unfolded.",
        "The situation feels bright and full of potential.",
        "I’m encouraged by how productive things have been.",
        "Everything feels peaceful and comfortable.",
        "I’m excited about the progress being made.",
        "This has been a refreshing and enjoyable time.",
        "I feel like everything is coming together nicely.",
        "The whole situation feels fair and uplifting.",
        "I’m energized by how well things are going.",
        "Everything about this feels right and satisfying.",
        "I feel hopeful about what comes next.",
        "This situation keeps lifting my spirits.",
        "I’m impressed by how smoothly everything is running.",
        "The outcome is far better than I expected.",
        "I feel motivated to keep moving forward.",
        "Everything feels lighter and more joyful.",
        "I’m proud of how this has turned out.",
        "The atmosphere feels calm and encouraging.",
        "I’m finding so many reasons to feel positive.",
        "Overall, this has been a happy and uplifting experience.","I feel deeply connected to the people around me.",
        "Every attempt to solve this seems to make it much simpler.",
        "I'm looking forward to what tomorrow might bring.",
        "The feedback I received was gentle and highly constructive.",
        "It feels like my hard work is being actively recognized.",
        "I'm finding it incredibly easy to focus on my tasks right now.",
        "The path ahead looks remarkably clear and certain.",
        "I feel like I'm making steady progress alongside my peers.",
        "The conversation left me feeling heard and understood.",
        "This project has become a massive source of inspiration for me.",
        "I feel very confident in my abilities lately.",
        "Everything I do seems to make a meaningful difference here.",
        "The energy in this room is incredibly revitalizing.",
        "I feel completely assured in every decision I make.",
        "My motivation has reached an all-time high today.",
        "It feels like we are moving forward with a clear goal.",
        "I am feeling very excited about the upcoming changes.",
        "The support I expected to have is readily available.",
        "This environment makes me feel valued and significant.",
        "The instructions are so clear that they are incredibly helpful.",
        "I feel like I'm investing my time wisely in this endeavor.",
        "The team seems to be pulling together in the same direction.",
        "I'm finding it very easy to stay patient with this process.",
        "The results we achieved are exceptional by any standard.",
        "I feel a profound sense of excitement with my current routine.",
        "This task is sparking a wealth of new creativity.",
        "I feel like an essential part of this group.",
        "The open communication is creating a lot of harmony.",
        "I am constantly confident that I will succeed.",
        "Everything feels cohesive and perfectly manageable.",
        "I feel like I'm being encouraged to do things I truly believe in.",
        "This place is filled with a sense of warmth and community.",
        "I clearly see the immense value in what I'm doing.",
        "The weight of the expectations feels entirely manageable.",
        "I feel like my opinions are being embraced with enthusiasm.",
        "Our daily output is substantial by any measure.",
        "I feel mentally free and expansive right now.",
        "This whole experience has left me feeling victorious.",
        "I feel like I can speak openly and freely around here.",
        "The future of this project looks incredibly bright.",
        "I am finding it easy to trust the people I am working with.",
        "The atmosphere here nurtures any attempt at innovation.",
        "I feel like a deeply valued contributor here.",
        "This situation is filling me with renewed vitality.",
        "The clear direction leaves me feeling focused.",
        "I am constantly finding reasons to believe in myself.",
        "I feel actively listened to during our meetings.",
        "The distribution of tasks is beautifully balanced.",
        "I am finding a deep sense of purpose in this work.",
        "I leave every day feeling completely energized.",
    ]

    train, test = train_test_split(positive_instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmful_instructions():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    """Load harmless instructions from Alpaca dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        item["instruction"] for item in dataset["train"] if item["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train[:512], test[:128]


@cache
def get_input_data(
    data_type: str, language_id: str = "en"
) -> Tuple[List[str], List[str]]:
    """Get training and test data."""
    if language_id != "en":
        raise ValueError(f"Only English (en) is supported, got: {language_id}")
    if data_type == "harmful":
        return get_harmful_instructions()
    elif data_type == "harmless":
        return get_harmless_instructions()
    elif data_type == "positive":
        return get_positive_instructions()
    elif data_type == "negative":
        return get_negative_instructions()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


# =============================================================================
# Tokenization and Hook Utilities
# =============================================================================


def tokenize_instructions_fn(instructions, tokenizer, system_prompt=None):
    """Tokenize instructions using chat template.

    Args:
        instructions: List of instruction strings
        tokenizer: HuggingFace tokenizer with chat template support
        system_prompt: Optional system prompt to prepend to each instruction

    Returns:
        Tokenized inputs as BatchEncoding with input_ids and attention_mask
    """
    inputs = tokenizer.apply_chat_template(
        [
            (
                [{"role": "user", "content": instruction}]
                if system_prompt is None
                else [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]
            )
            for instruction in instructions
        ],
        padding=True,
        truncation=False,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


@contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
    module_forward_hooks: List[Tuple[nn.Module, Callable]] = None,
    **kwargs,
):
    """Context manager for temporarily adding forward hooks.

    Args:
        module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
        module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
        **kwargs: Additional keyword arguments passed to hook functions

    Yields:
        None. Hooks are active within the context, removed on exit.
    """
    module_forward_pre_hooks = module_forward_pre_hooks or []
    module_forward_hooks = module_forward_hooks or []
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


# =============================================================================
# Activation Capture Hooks
# =============================================================================


def get_residual_hook(
    cache_key_prefix: str, cache: dict, positions: list, extract_positions: list = None
):
    """Create hook for decoder layers to capture residual stream at resid_post.

    Args:
        cache_key_prefix: Prefix for cache keys (e.g., 'layer_10')
        cache: Dictionary to store captured activations
        positions: List of positions to capture - typically ['post'] for end of layer
        extract_positions: Token positions to extract (default [-1] for last token)

    Returns:
        Hook function that captures layer outputs to cache
    """
    extract_positions = extract_positions or [-1]

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output

        if "post" in positions:
            cache_key = f"{cache_key_prefix}_post"
            if extract_positions == [-1]:
                acts = hidden_states[:, -1:, :].detach().cpu()
            else:
                acts = hidden_states[:, extract_positions, :].detach().cpu()

            if cache_key in cache:
                cache[cache_key] = torch.cat([cache[cache_key], acts], dim=0)
            else:
                cache[cache_key] = acts

    return hook_fn


def get_mlp_input_hook(cache_key: str, cache: dict, extract_positions: list = None):
    """Create pre-hook to capture resid_mid from post_attention_layernorm input.

    Args:
        cache_key: Key to use in cache dictionary (e.g., 'layer_10_mid')
        cache: Dictionary to store captured activations
        extract_positions: Token positions to extract (default [-1] for last token)

    Returns:
        Pre-hook function that captures layer inputs to cache
    """
    extract_positions = extract_positions or [-1]

    def hook_fn(module, input):
        if extract_positions == [-1]:
            acts = input[0][:, -1:, :].detach().cpu()
        else:
            acts = input[0][:, extract_positions, :].detach().cpu()

        if cache_key in cache:
            cache[cache_key] = torch.cat([cache[cache_key], acts], dim=0)
        else:
            cache[cache_key] = acts

    return hook_fn
