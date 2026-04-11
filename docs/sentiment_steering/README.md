# Sentiment Steering — Template Discovery Notes

Session notes for transferring working knowledge to a new environment (e.g.,
a machine with larger GPUs). If you are a fresh Claude Code session reading
this, start here before making changes to sentiment-steering code.

## What this session accomplished

- Built an iterative template-discovery pipeline for sentiment angular steering
- Found that **`echo_en` is the best-performing template on Qwen2.5-7B** with a
  ~300° clean steering range (out of 360°) under the REP + CN + EMPTY flag
- Identified catastrophic failure of the legacy `restate` template on 7B
  (68% repetition at 120°), mirroring the same catastrophic-zone pattern
  previously seen on 14B and 32B
- Ran full multi-template sweeps (4 templates × 24 conditions × 3,071 tweets =
  310,171 rows) on 7B and 3B with `run_sentiment_experiment.py`
- Documented a reproducible template-discovery workflow that can be re-run on
  any model for which sentiment directions have been extracted

## Files added in this session

```
run_template_discovery.py          # iterative template × angle sweep
analyze_template_discovery.py      # heatmap + clean-range scoring
run_sentiment_experiment.py        # MODIFIED: now supports --templates CSV list
docs/sentiment_steering/README.md  # this file
```

## Quick-start recipes

All commands from repo root. Set `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
(the scripts do this automatically).

### 1. Template discovery on a new model

Run a round of candidate templates on a ~150-tweet balanced subset to find
the widest "clean steering range" (the span of angles where degeneration
rate stays below 10%):

```bash
python run_template_discovery.py --model 7B --round 1 --adaptive-mode 0
python analyze_template_discovery.py results/template_discovery_7B_round1_mode0.csv
```

Edit `TEMPLATES_BY_ROUND` in `run_template_discovery.py` to define new
rounds. Each round is a dict of `name -> prompt template string` using
`{tweet}` as the placeholder. Round 1 has the "standard" 6 candidates; rounds
2 and 3 in the file are historical iterations (kept for reference).

Iterate by adding Round N+1 with refined candidates and re-running.

### 2. Full sentiment experiment with best templates

Once you have a template set that exceeds ~270° clean range, run the full
angular + CAA + prompted sweep on 3,071 TSAD tweets:

```bash
python run_sentiment_experiment.py --model 7B \
    --templates restate,echo_en,similar_tweet_en,rewrite
```

Output: `results/results_<MODEL>_<DATE>_<template1>_<template2>_..._.csv`.
Columns: `model, tweet_id, original_text, ground_truth, method, param_name,
param_value, prompt_template, generated_text`. Prompted sentiment runs once
(it has its own template); baseline + angular + CAA run once per template.

Dry-run first to confirm row counts:
```bash
python run_sentiment_experiment.py --model 7B --templates echo_en --dry-run
```

### 3. Applying to a new larger model

```bash
# 1. Ensure sentiment directions are extracted — must include max_sim
ls output/<ModelName>/STMT/steering_config-en-*max_sim*.npy
# If missing:
cd pytorch_pure && python extract_directions_sentiment.py --model <HF_model_id>
cd ..

# 2. Register the model in MODEL_CONFIGS in both:
#    - run_template_discovery.py
#    - run_sentiment_experiment.py
#    Point at the max_sim STMT config by default.

# 3. Run discovery to pick winners for this model:
python run_template_discovery.py --model <NEW> --round 1

# 4. Full sweep with winners
python run_sentiment_experiment.py --model <NEW> --templates <best_templates>
```

## Key findings (with evidence)

### The current winning template is `echo_en`

```
Tweet: '{tweet}'
The same thing expressed in different English words:
```

Clean-range measurements on Qwen2.5-7B (mode 0, max_sim_19_mid config):
- **Template discovery (15° steps, 150 tweets):** 285° clean range (R1 winner)
- **Full sweep (30° steps, 3,071 tweets):** 300° clean range
- Only 120° (12.2%) and 150° (15.4%) exceed 10% flagged
- Every other angle is <7% on 7B

### The 120°–150° transition zone resists prompting

All six Round 1 templates fail in this zone; 11 Round 2/3 variants
(quoted strings, few-shot, third-person, anti-list instructions, newline
scaffolding) **none** beat the original `echo_en` on clean range. The pattern
is fundamental to the steering direction rotating through a region where
next-token distributions collapse into repetition loops (observed output:
`1. ... 2. ... 3. ...` lists and `Feeling: Feeling: Feeling:` nested headers).

Templates that improve 120°–150° tend to **break 180°–210° in trade** —
the bad zone can be slid around the circle but not eliminated by prompting
alone. Interventions to try if you want to fix this:
- Swap steering config (`max_norm` instead of `max_sim` or vice versa)
- Measure activation norms per angle to see where the manifold breaks
- Template ensemble (different template per angle band) — untested

### Adaptive mode 1 HURTS `echo_en`

Mode 0 (always steer): **285°** clean range
Mode 1 (adaptive, only steer when aligned): **240°** clean range

Intuition: adaptive mode selectively reduces steering strength, which for
`echo_en` apparently pulls activations *closer* to the broken zone rather than
away. Default to **mode 0** unless you have a specific reason.

### Scale dramatically affects template winners

| Model | config used | Best template | Worst template |
|-------|---|---|---|
| 3B | max_sim_27_mid **L20-30** subset | **rewrite** (0.4% mean) | echo_en (15% mean) |
| 7B | max_sim_19_mid (all layers) | **echo_en** (4.9% mean) | rewrite (14.3% mean) |
| 14B | max_sim_36_mid | ? (not yet rerun with echo_en) | restate (72% @ 270°) |
| 32B | max_sim_47_mid | ? (not yet rerun with echo_en) | restate (broad CN leaks) |

**Critical:** Do not assume `echo_en` will transfer to larger models without
re-running discovery. The best template inverts between 3B and 7B —
on 3B `echo_en` fails at baseline (list-expansion lock-in from the trailing
colon), and on 7B `rewrite` fails at 120°-180° (catastrophic repetition at
the sentiment transition zone). The winning template is model-specific.

The most cross-model-consistent template is `similar_tweet_en` — 210° clean
on 7B and 360° clean on 3B. Not the best on either, but the only one without
a catastrophic failure somewhere.

### 3B needs layer-subset steering (not all-layer)

Applying angular steering at every layer of Qwen2.5-3B destroys outputs
(42-99% degenerate across angles), even though CAA with the same direction
is completely clean. The problem is the angular rotation math: it *removes*
the projection of the activation onto the {b1, b2} plane and replaces it
with a unit vector in direction v_theta. Repeating this rotation at 71
module entries (36 layers × 2 layernorm modules each) compounds into
coherence destruction.

The fix is to apply steering only at a **subset of layers** near the peak
cosine-similarity layer. On 3B, `filter_config_layers.py` was used to
produce:
- `max_sim_27_mid_L27only` (1 layer, 2 modules) — clean but weak (62-85%
  steering effect)
- `max_sim_27_mid_L25-29`  (5 layers, 10 modules) — clean, subtle shifts
- **`max_sim_27_mid_L20-30` (11 layers, 22 modules)** — clean, real
  sentiment flips (95-100% effect), now the default 3B config

On 7B, the full-layer config works fine because 7B's sentiment direction is
more consistent across layers (higher cosine). Layer-subset filtering is
specific to small models where the direction doesn't align cleanly.

### Failure-mode catalog

Seen across models and templates:
- **Repetition loops (REP):** `1. Shanghai... 2. Shanghai... 3. Shanghai...`,
  `and and and and`, `Feeling: Feeling: Feeling:`. Dominant failure on 7B.
- **Chinese character leaks (CN):** Random CJK tokens in output. Dominant
  failure on 14B/32B with `restate`, mostly absent on 7B with English-locked
  templates.
- **Empty outputs (EMPTY):** Model generates <10 chars. Rare across the board.
- **Off-topic drift (OFFTOPIC):** Low Jaccard overlap with input tweet.
  Tracked as secondary metric only, not the primary optimization target.

Scoring metric (from `analyze_template_discovery.py`):
Primary flag rate = fraction of rows with any of {REP, CN, EMPTY}.
Clean angle = primary flag rate < 10% at that angle.
Clean range = total angular span (in degrees) of clean angles.

## Gotchas and setup requirements

### Must-have environment

- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (scripts set this)
- `enforce_eager=True` on the vLLM `LLM()` (required for hook registration)
- Single-GPU default; for models that don't fit at full precision, cap
  `max_model_len` (not quantize — user preference)

### The 3B sentiment direction issue

`output/Qwen2.5-3B-Instruct/STMT/` only contains
`steering_config-en-max_norm_35_post-pca_0.npy`. There is **no max_sim config
for 3B sentiment** in the repo. Results from the 3B run use `max_norm` and
show much weaker steering (64–93% of outputs differ between opposite angles
vs. 100% on 7B). If you want apples-to-apples 3B comparison, re-extract:

```bash
cd pytorch_pure
python extract_directions_sentiment.py --model Qwen/Qwen2.5-3B-Instruct
```

This should produce a `max_sim` variant alongside the existing max_norm.

### `run_sentiment_experiment.py` append mode

The script appends to its output CSV rather than overwriting. If you re-run
with the same model + date + templates, rows will accumulate (2x duplication,
as previously seen with 32B). Either delete the target file first or verify
you want to append. The script now prints a warning if the output file exists.

### Auto-detect config prefers max_sim

`run_sentiment_experiment.py` has auto-detect logic that prefers
`max_sim` over `max_norm` when multiple configs exist in the STMT directory.
Override with `--config <path>` if you need a specific file.

## Available templates in the steering scripts

Defined in `STEERING_TEMPLATES` in `run_sentiment_experiment.py`:

| Name | Template |
|---|---|
| `restate` | `Analyze the situation described in this tweet and restate the core event in a single sentence: '{}'\nSingle sentence summary:` |
| `echo_en` | `Tweet: '{}'\nThe same thing expressed in different English words:` |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{}'\nSimilar tweet:` |
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{}'\nRewritten tweet:` |

To add a new template: update `STEERING_TEMPLATES` in
`run_sentiment_experiment.py` (add it to `TEMPLATES_BY_ROUND` in
`run_template_discovery.py` for discovery).

## Anti-patterns (things that made things worse)

From Round 2 and Round 3 of template discovery on 7B:

- **"Do not make a list" explicit instruction** — paradoxically primes list
  output (`echo_one_sentence` had 46.7% flagged at 135°)
- **Few-shot priming** — the model locks onto the demonstrated format and
  collapses faster under steering (`echo_fewshot` 48% at 150°)
- **Newline scaffolding before completion** — broke baseline itself
  (`echo_prefix` had 24.7% at baseline)
- **Third-person framing** — didn't help either direction, and introduced
  "Feeling: Feeling:" header loops (`echo_third_person` 32% at 165°)

Templates that did *not* make it worse but didn't beat `echo_en`:
`echo_quoted` (270° clean, but trades 150° for 180°), `echo_rephrased` (240°),
`echo_is` (240°), `echo_brief` (240°).

## Next steps if you want to push further

1. **Fix 3B by extracting max_sim sentiment directions** (see Gotchas).
2. **Try the full sweep on 14B/32B with `echo_en`** — this should dramatically
   improve over the original `restate` runs (14B restate peaked at 72% flagged
   at 270°; `echo_en` on 7B was 3%). The prior 14B/32B result CSVs are in
   `results/results_14B_20260405.csv` and `results/results_32B_20260405.csv`
   for comparison.
3. **Investigate the 120°–150° zone structurally** — measure activation norms
   and cosine similarity between rotated activations and the training
   distribution. This is the only failure pattern prompting hasn't fixed.
4. **Template ensemble** — route tweets through different templates depending
   on the target angle. Unexplored.
5. **Compare adaptive modes per template** — mode 0 beat mode 1 for `echo_en`
   but this isn't guaranteed for other templates.

## Result CSVs in the repo

Under `results/`:

- `results_7B_20260411_restate_echo_en_similar_tweet_en_rewrite.csv.gz` —
  **Flagship 7B run.** Full sweep with 4 templates (310k rows gzipped),
  using `max_sim_19_mid` config (all layers). Decompress with
  `gunzip -k results/results_7B_20260411_restate_echo_en_similar_tweet_en_rewrite.csv.gz`
- `results_3B_20260411_restate_echo_en_similar_tweet_en_rewrite.csv.gz` —
  **3B L20-30 run.** Full sweep with 4 templates (310k rows gzipped),
  using the L20-30 filtered config (`max_sim_27_mid_L20-30-pca_0.npy`).
  rewrite is the best template on 3B (0.4% mean flagged, 360° clean).
- `template_discovery_7B_round{1,2,3}_mode{0,1}_scores.csv` — per-round
  template × angle degeneration summary statistics (the analysis output of
  `analyze_template_discovery.py`). These small score files justify the
  `echo_en` winner choice on 7B.

**Intentionally not committed** (regeneratable or redundant):
- Raw template discovery CSVs (~33 MB each) — regenerate with
  `python run_template_discovery.py --model 7B --round N`
- `results_7B_20260411.csv` (the echo_en-only run, 39 MB) — redundant with
  the multi-template gzip above (same conditions, different stochastic outputs)
- Earlier 3B runs with the wrong configs (`max_norm_35_post` was too weak,
  `max_sim_27_mid` full-layers destroyed outputs) — the L20-30 run is the
  correct one
- All per-round log files
