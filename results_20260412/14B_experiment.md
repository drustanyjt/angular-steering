# 14B bf16 Sentiment Steering Sweep — 2026-04-12

## Setup

- **Model:** Qwen2.5-14B-Instruct (bf16, no quantization)
- **GPU:** NVIDIA H100 80GB HBM3
- **Steering config:** `max_sim_36_mid` (all 48 layers, max-cosine-similarity at layer 36 mid-position)
- **Angular step:** 15° (24 angles)
- **Dataset:** 3,071 TSAD tweets (962 pos, 900 neg, 1,209 neu)
- **Conditions:** 185 (36 per template x 5 templates + 5 prompted)
- **Total rows:** 568,135
- **Runtime:** ~64 min on H100

## Templates used

Five templates selected for cross-comparison: three clean paraphrase-style
templates, one emotion-framing template, and one cross-model baseline.

| Name | Prompt | Rationale |
|---|---|---|
| `rewrite_en` | `Rewrite this tweet in English using different words: '{tweet}'\nRewritten tweet (English):` | Cleanest 90° neutralization; best CN leak prevention |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{tweet}'\nSimilar tweet:` | Most natural tweet format; cross-model baseline |
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{tweet}'\nRewritten tweet:` | Clean paraphrase; comparison to rewrite_en |
| `similar_equivalent_en` | `Write a similar English tweet with an equivalent emotional reaction: '{tweet}'\nSimilar tweet:` | Emotion-framing amplifies sentiment flips (~60% more flips than paraphrase templates) |
| `echo_en` | `Tweet: '{tweet}'\nThe same thing expressed in different English words:` | 7B winner (300° clean); included to test cross-scale transfer (degrades to 90° on 14B) |

## Prompt engineering insights

- **Emotion framing amplifies steering:** Templates with "emotional reaction" wording give angular steering a handle to manifest as observable sentiment change. `similar_equivalent_en` got 862/3,071 flips (28%) vs ~540 (18%) for paraphrase templates — same clean range.
- **"English" in the template reduces CN leaks:** `rewrite_en` vs `rewrite` cuts Chinese character leakage at 105°-165°.
- **90° steering neutralizes:** At 90° (perpendicular to sentiment axis), negative tweets get softened ("Life sucks" → "life can be challenging") and positive tweets get toned down ("was a success" → "went well"). Effect is subtle but consistent and distinct from unsteered baseline.
- **0° positive pushing is noisy:** Positive-direction steering often produces hashtag spam and emoji floods rather than coherent positive text. 180° negative pushing produces more natural empathetic text (likely RLHF alignment).
- **Templates don't transfer across scales:** `echo_en` is the 7B winner (300° clean) but collapses to 90° on 14B. `rewrite_en` is the best at >=14B scale.

## Results summary (strict REP metric)

| Template | Clean range | Flips/3071 | Flip% | Mean delta |
|---|---|---|---|---|
| similar_equivalent_en | 150° | 862 | 28% | -3.82 |
| echo_en | 90° | 565 | 18% | -2.41 |
| similar_tweet_en | 150° | 543 | 18% | -3.25 |
| rewrite_en | 150° | 540 | 18% | -3.39 |
| rewrite | 135° | 516 | 17% | -3.23 |

## Failure modes

- **REP (repetition):** Dominant at 225°-345° (positive-pushing side). Hashtag soup, "I'm so cool!" loops.
- **CN (Chinese characters):** Dominant at 105°-165° (transition zone). Up to 28% for rewrite_en.
- **Meta-explanation:** Model drops into instruction-following mode ("Step 1: Identify the emotion...") instead of generating a tweet. Affects ~30% of 180° outputs.

## Command used

```bash
python run_sentiment_experiment.py --model 14B \
    --templates rewrite_en,similar_tweet_en,rewrite,similar_equivalent_en,echo_en \
    --angular-step 15 \
    --config output/Qwen2.5-14B-Instruct/STMT/steering_config-en-max_sim_36_mid-pca_0.npy
```
