# 32B bf16 Sentiment Steering Sweep — 2026-04-12

## Setup

- **Model:** Qwen2.5-32B-Instruct (bf16, no quantization)
- **GPU:** NVIDIA H100 80GB HBM3 (74.6 GB VRAM used, 100% GPU utilization)
- **Steering config:** `max_sim_47_mid` (all 64 layers, max-cosine-similarity at layer 47 mid-position)
- **Angular step:** 15° (24 angles)
- **Dataset:** 3,071 TSAD tweets (962 pos, 900 neg, 1,209 neu)
- **Conditions:** 185 (36 per template x 5 templates + 5 prompted)
- **Total rows:** 568,135
- **Runtime:** ~190 min on H100 (~62s/condition)
- **Runtime:** ~175 min on H100

## Templates used

Same five templates as the 14B experiment for direct cross-scale comparison.

| Name | Prompt | Rationale |
|---|---|---|
| `rewrite_en` | `Rewrite this tweet in English using different words: '{tweet}'\nRewritten tweet (English):` | Best on 32B for CN leak prevention; cleanest neutralization at 90° |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{tweet}'\nSimilar tweet:` | Cross-model baseline; sometimes overshoots to positive at 90° |
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{tweet}'\nRewritten tweet:` | Clean paraphrase; comparison point |
| `similar_equivalent_en` | `Write a similar English tweet with an equivalent emotional reaction: '{tweet}'\nSimilar tweet:` | Cross-model winner from R7-R12 discovery; consistent 38-41 flips across all 4 scales |
| `echo_en` | `Tweet: '{tweet}'\nThe same thing expressed in different English words:` | 7B winner; included to confirm cross-scale degradation |

## Prompt engineering insights

- **32B has catastrophic CN leakage:** At 165°-270°, the model collapses into Chinese characters (up to 99% for `rewrite` template). `rewrite_en` with explicit "English" framing cuts this but can't eliminate it.
- **32B failure topology is phase-shifted from 14B:** REP dominates at 30°-165° (vs 225°-345° on 14B), CN at 165°-270° (vs 105°-165° on 14B). The clean windows are on opposite sides of the circle.
- **Emotion templates have scale-specific winners:** On 32B discovery, `similar_matching_en` got 77/150 flips (51%) vs `similar_equivalent_en`'s 38 (25%). But `similar_equivalent_en` is the most consistent across all four model sizes.
- **135° clean range ceiling holds on 32B** regardless of template choice.

## Results summary (strict REP metric)

| Template | Clean range | Flips/3071 | Flip% | Mean delta |
|---|---|---|---|---|
| similar_equivalent_en | 135° | 930 | 30% | -1.91 |
| similar_tweet_en | 135° | 569 | 19% | -1.34 |
| rewrite_en | 150° | 446 | 15% | -1.07 |
| echo_en | 120° | 360 | 12% | -0.87 |
| rewrite | 150° | 86 | 3% | -1.07 |

`similar_equivalent_en` dominates with 930 flips (30%) — 63% more than the
next best (`similar_tweet_en` at 569). The emotion framing effect is even
stronger on 32B than on 14B (930 vs 862 flips). `rewrite` nearly suppresses
sentiment flips entirely (3%) despite having 150° clean range — confirming
that paraphrase-only templates absorb steering silently.

## Failure modes

- **CN (Chinese characters):** Dominant at 165°-270°. Up to 99% for `rewrite` at 195°. `rewrite_en` cuts this substantially but still peaks at 98% at 225°.
- **REP (repetition):** Dominant at 30°-165°. Up to 93% for `echo_en` at 90°.
- **Meta-explanation:** Similar to 14B — model drops into instruction mode at 180°.

## Command used

```bash
python run_sentiment_experiment.py --model 32B \
    --templates rewrite_en,similar_tweet_en,rewrite,similar_equivalent_en,echo_en \
    --angular-step 15 \
    --config output/Qwen2.5-32B-Instruct/STMT/steering_config-en-max_sim_47_mid-pca_0.npy
```
