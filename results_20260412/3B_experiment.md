# 3B Sentiment Steering Sweep — 2026-04-12

## Setup

- **Model:** Qwen2.5-3B-Instruct (bf16)
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB)
- **Steering config:** `max_sim_27_mid` **layer-subset L20-30** (11 of 36 layers, 22 module hooks)
- **Angular step:** 15° (24 angles)
- **Dataset:** 3,071 TSAD tweets (962 pos, 900 neg, 1,209 neu)
- **Total rows:** 789,247

## Why layer-subset steering on 3B

Applying angular steering at all 36 layers destroys 3B outputs (42-99%
degenerate across all angles). CAA with the same direction is clean,
confirming the direction is valid — the problem is the angular rotation
math compounding through too many layers.

The fix: apply steering only at layers 20-30 (around the peak-cosine
layer 27). This produces clean outputs with real sentiment flips.
See `filter_config_layers.py` for the tool used to create the subset.

| Config | Layers | Clean range | Sentiment flips |
|--------|--------|-------------|-----------------|
| max_sim_27_mid full (36 layers) | 71 modules | 0° (all destroyed) | 100% different but incoherent |
| max_sim_27_mid L27only (1 layer) | 2 modules | 360° | 62-85% different but no sentiment flip |
| max_sim_27_mid L25-29 (5 layers) | 10 modules | 360° | 87-99% different, subtle shifts |
| **max_sim_27_mid L20-30 (11 layers)** | **22 modules** | **360°** | **95-100% different, real sentiment flips** |

## Templates

Same 7 templates as the 7B experiment for direct cross-model comparison.

### Emotion-framing templates

| Name | Prompt | 3B flip rate | 3B clean range |
|---|---|---|---|
| `similar_equivalent_en` | `Write a similar English tweet with an equivalent emotional reaction: '{tweet}'\nSimilar tweet:` | 23% (cross-model winner, 34/150 on discovery set) | 360° |
| `similar_parallel_en` | `Write a similar English tweet with a parallel emotional reaction: '{tweet}'\nSimilar tweet:` | 11% (best on 7B but weaker on 3B) | 360° |
| `similar_reaction_en` | `Write a similar English tweet with a similar emotional reaction: '{tweet}'\nSimilar tweet:` | 8% (R9 breakthrough template) | 360° |

### Clean paraphrase templates

| Name | Prompt | 3B flip rate | 3B clean range |
|---|---|---|---|
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{tweet}'\nRewritten tweet:` | 2% (cleanest: 0.1% mean flagged, but almost no sentiment flip) | 360° |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{tweet}'\nSimilar tweet:` | 4% (cross-model baseline) | 360° |
| `restate` | `Analyze the situation described in this tweet and restate the core event in a single sentence: '{tweet}'\nSingle sentence summary:` | 2% | 360° |
| `echo_en` | `Tweet: '{tweet}'\nThe same thing expressed in different English words:` | 2% (broken on 3B baseline at 3.2% due to list-induction from trailing colon — 3B interprets `:` as "start numbered list") | 360° |

### 3B-specific notes

- **`echo_en` fails on 3B baseline** (3.2% flagged even without steering)
  because 3B interprets the trailing colon format as "produce a numbered
  list". All other templates with colons after action words ("Rewritten
  tweet:", "Similar tweet:") are fine — only the noun-phrase colon
  ("different English words:") triggers list-induction.
- **`rewrite` is the cleanest but weakest** — 0.1% degeneration but only
  2% sentiment flips. The steering is rotating activations but the model
  just produces a faithful paraphrase.
- **Best template per model is different:** 3B prefers `rewrite` for
  cleanliness or `similar_equivalent_en` for flips. 7B prefers `echo_en`
  for cleanliness or `similar_parallel_en` for flips.

## Per-template conditions

Identical to 7B: 7 templates × 36 conditions + 5 prompted = 789,247 rows.
See 7B_experiment.md for column schema.

## Command

```bash
# Original 4 templates
python run_sentiment_experiment.py --model 3B \
    --templates restate,echo_en,similar_tweet_en,rewrite --angular-step 15 \
    --config output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid_L20-30-pca_0.npy

# New 3 emotion-framing templates
python run_sentiment_experiment.py --model 3B \
    --templates similar_equivalent_en,similar_parallel_en,similar_reaction_en \
    --angular-step 15 \
    --config output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid_L20-30-pca_0.npy
```

Results concatenated with duplicate prompted rows removed.
