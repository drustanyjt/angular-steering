# 7B Sentiment Steering Sweep — 2026-04-12

## Setup

- **Model:** Qwen2.5-7B-Instruct (bf16)
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB)
- **Steering config:** `max_sim_19_mid` (all 28 layers)
- **Angular step:** 15° (24 angles)
- **Dataset:** 3,071 TSAD tweets (962 pos, 900 neg, 1,209 neu)
- **Total rows:** 789,247

## Templates

Seven templates covering two categories: **emotion-framing** (amplify
sentiment shifts) and **clean paraphrase** (minimize degeneration).

### Emotion-framing templates (discovered R7-R12)

These give angular steering a "handle" to flip sentiment. The explicit
mention of emotion in the prompt causes the model to generate
emotion-labelled text, which the steering rotation then shifts.

| Name | Prompt | Flip rate | Clean range |
|---|---|---|---|
| `similar_equivalent_en` | `Write a similar English tweet with an equivalent emotional reaction: '{tweet}'\nSimilar tweet:` | 27% (cross-model winner from R11, most consistent across 3B/7B/14B/32B) | 300° |
| `similar_parallel_en` | `Write a similar English tweet with a parallel emotional reaction: '{tweet}'\nSimilar tweet:` | 31% (best 7B flip count: 46/150) | 315° |
| `similar_reaction_en` | `Write a similar English tweet with a similar emotional reaction: '{tweet}'\nSimilar tweet:` | 27% (R9 breakthrough that started the emotion-framing discovery) | 315° |

### Clean paraphrase templates

These produce clean rephrased text but suppress the sentiment shift
(~2-12% flip rate vs 27-31% for emotion-framing). Included for baseline
comparison and for use cases where output quality matters more than
steering visibility.

| Name | Prompt | Flip rate | Clean range |
|---|---|---|---|
| `echo_en` | `Tweet: '{tweet}'\nThe same thing expressed in different English words:` | 12% (7B-specific winner, 360° clean but broken on 3B baseline) | 360° |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{tweet}'\nSimilar tweet:` | 9% (cross-model baseline, predecessor to emotion-framing variants) | 285° |
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{tweet}'\nRewritten tweet:` | 10% (direct imperative, cleanest on 3B but weak flips) | 285° |
| `restate` | `Analyze the situation described in this tweet and restate the core event in a single sentence: '{tweet}'\nSingle sentence summary:` | 6% (analytical extraction, catastrophic at 120° on 7B) | 285° |

### Key insight

Templates without emotion-related language produce ~10x fewer sentiment
flips. The steering vector rotates the same activations regardless of
template, but the prompt controls whether the model *expresses* the
shifted emotion or silently absorbs it back into a neutral paraphrase.

## Per-template conditions

Each of the 7 steering templates gets:
- 1 baseline (no steering)
- 24 angular angles (0°, 15°, ..., 345°) at adaptive_mode=0
- 11 CAA alphas (-10, -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5, 10)
= 36 conditions × 3,071 tweets = 110,556 rows per template

Plus 5 prompted-sentiment baselines (3,071 each) using their own template:
`"Rewrite this tweet with a {tone} tone: '{tweet}'\nRewritten tweet:"`

Total: 7 × 110,556 + 5 × 3,071 = 789,247 rows.

## CSV columns

```
model, tweet_id, original_text, ground_truth, method, param_name,
param_value, prompt_template, generated_text
```

- `method`: baseline | angular | caa | prompted
- `param_value`: angle in degrees (angular), alpha float (caa), sentiment label (prompted), or "0" (baseline)
- `prompt_template`: template name from tables above, or `prompted_{sentiment}`

## Command

```bash
# Original 4 templates
python run_sentiment_experiment.py --model 7B \
    --templates restate,echo_en,similar_tweet_en,rewrite --angular-step 15

# New 3 emotion-framing templates (appended, prompted stripped)
python run_sentiment_experiment.py --model 7B \
    --templates similar_equivalent_en,similar_parallel_en,similar_reaction_en \
    --angular-step 15
```

Results concatenated into a single CSV with duplicate prompted rows removed.
