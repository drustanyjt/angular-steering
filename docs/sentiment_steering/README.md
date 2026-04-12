# Sentiment Steering — Template Discovery Notes

Session notes for transferring working knowledge to a new environment (e.g.,
a machine with larger GPUs). If you are a fresh Claude Code session reading
this, start here before making changes to sentiment-steering code.

## TL;DR for H100 handoff

We have clean flagship runs on 7B and 3B with angular steering. 14B and 32B
are the next targets. **14B fp8 attempts on a 24 GB GPU were inconclusive**
(see dedicated section below) — re-run them on H100 with bf16 to get a
definitive answer.

Start with:
```bash
bash setup_and_run.sh 14B restate,echo_en,similar_tweet_en,rewrite 15
bash setup_and_run.sh 32B rewrite_en,echo_en,similar_tweet_en,rewrite 15
```

(The 32B template set swaps `restate` for `rewrite_en` because prior 32B
discovery showed `restate` has the same catastrophic 270° failure as on 14B
while `rewrite_en` cut Chinese-character leakage significantly.)

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

## Pipeline file layout

```
sentiment_pipeline.py              # SHARED module: TEMPLATES, MODEL_CONFIGS,
                                   # load_tsad_tweets, ResultWriter, build_llm,
                                   # CSV_COLUMNS. One source of truth for
                                   # everything that used to be duplicated
                                   # between the two run scripts.
run_template_discovery.py          # Iterative template × angle sweep on a
                                   # balanced ~150-tweet subset. Uses rounds
                                   # (TEMPLATES_BY_ROUND) to organize
                                   # template candidates across iterations.
run_sentiment_experiment.py        # Full sweep on 3,071 TSAD tweets with
                                   # baseline + angular + CAA + prompted
                                   # per template.
analyze_template_discovery.py      # Heatmap + clean-range scoring. Handles
                                   # both canonical and legacy CSV schemas,
                                   # reads .csv or .csv.gz transparently.
filter_config_layers.py            # Filter a steering config .npy to keep
                                   # only a subset of layer modules (needed
                                   # for 3B — all-layer angular destroys it).
pytorch_pure/extract_directions_sentiment.py
                                   # Extracts sentiment directions from a
                                   # model using 80 positive + 80 negative
                                   # hardcoded instruction pairs.
docs/sentiment_steering/README.md  # this file
```

**Single CSV schema (canonical):**
```
model, tweet_id, original_text, ground_truth,
method, param_name, param_value, prompt_template, generated_text
```
Both `run_sentiment_experiment.py` and `run_template_discovery.py` write
this schema via `sentiment_pipeline.ResultWriter`. The analyzer auto-detects
legacy discovery CSVs that use `template`/`angle`/`text` columns, so old
runs still score.

## Quick-start recipes

All commands from repo root. Scripts set `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
automatically. Both experiment scripts import shared constants from
`sentiment_pipeline.py` (templates, model configs, data loading, CSV writer).

### 1. Template discovery on a new model

Run a round of candidate templates on a balanced 150-tweet subset to find
the widest "clean steering range" (span of angles where REP+CN+EMPTY < 10%):

```bash
python run_template_discovery.py --model 7B --round 1 --adaptive-mode 0
python analyze_template_discovery.py \
    results/template_discovery_7B_<date>_round1_mode0.csv
```

Rounds are defined in `TEMPLATES_BY_ROUND` inside `run_template_discovery.py`.
Base templates live in `sentiment_pipeline.TEMPLATES`; ad-hoc experimental
variants that only appear in later rounds live inline in `_AD_HOC`.

Iterate by adding Round N+1 (historical rounds 1-6 are kept for reference)
and rerunning.

### 2. Full sentiment experiment with best templates

Once you have a template set that exceeds ~270° clean range, run the full
angular + CAA + prompted sweep on 3,071 TSAD tweets:

```bash
python run_sentiment_experiment.py --model 7B \
    --templates restate,echo_en,similar_tweet_en,rewrite
```

Optional flags:
- `--angular-step 15` → 24 angles instead of 12 (filename gets `_15deg` tag)
- `--adaptive-mode 1` → only steer when activation aligns with direction
- `--config path/to/steering.npy` → override the default config
- `--dry-run` → print row counts without loading the model

Output filename format:
`results/results_<MODEL>_<DATE>[_<step>deg]_<template1>_<template2>_..._.csv`

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

# 2. Register the model in MODEL_CONFIGS (sentiment_pipeline.py).
#    Both run scripts read from this single dict. For models that can't
#    fit at bf16 on the target GPU, add "llm_kwargs": {"quantization": "fp8"}
#    (or similar) and optionally "max_model_len".

# 3. Run discovery to pick winners for this model:
python run_template_discovery.py --model <NEW> --round 1

# 4. Full sweep with winners
python run_sentiment_experiment.py --model <NEW> --templates <best_templates>
```

### 4. Analyzing results from any run

```bash
# Works on template discovery CSVs (balanced subset, 150 tweets)
python analyze_template_discovery.py results/template_discovery_7B_round1_mode0.csv

# Works on full experiment CSVs (3,071 tweets) — including gzipped
python analyze_template_discovery.py \
    results/results_7B_20260412_15deg_restate_echo_en_similar_tweet_en_rewrite.csv.gz
```
The analyzer auto-detects the CSV schema and supports `.csv.gz` transparently.
It prints a heatmap, per-flag breakdown, and a template summary ranked by
widest circular-contiguous clean angle range.

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

`sentiment_pipeline.resolve_config_path` prefers the newest `max_sim`
config in `output/<model_name>/STMT/` when `MODEL_CONFIGS[key]["config"]`
is None. Override with `--config <path>` if you need a specific file.

## Available templates in the steering scripts

Defined in `sentiment_pipeline.TEMPLATES` (single source of truth):

| Name | Template |
|---|---|
| `restate` | `Analyze the situation described in this tweet and restate the core event in a single sentence: '{tweet}'\nSingle sentence summary:` |
| `rewrite` | `Rewrite this tweet to say the same thing in different words: '{tweet}'\nRewritten tweet:` |
| `rewrite_en` | `Rewrite this tweet in English using different words: '{tweet}'\nRewritten tweet (English):` |
| `paraphrase_en` | `Paraphrase this tweet in English: '{tweet}'\nParaphrase:` |
| `similar_tweet_en` | `Write a similar tweet in English about the same topic: '{tweet}'\nSimilar tweet:` |
| `echo_en` | `Tweet: '{tweet}'\nThe same thing expressed in different English words:` |

To add a new persistent template, edit `sentiment_pipeline.TEMPLATES`.
One-off experimental variants that are only used for a specific discovery
round live in `_AD_HOC` inside `run_template_discovery.py`.

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

## 14B fp8 — inconclusive findings (handoff for H100)

Attempted on a 24 GB GPU with `--model 14B_fp8` (vLLM dynamic FP8). **No
configuration gave both clean outputs AND meaningful sentiment flips
simultaneously**, so these results are *not* the flagship-quality data the
7B and 3B runs produced. Reopen this on an H100 and re-run with the full
bf16 model to get a clean answer.

### What I tried (all on Qwen2.5-14B-Instruct fp8, 150 curated tweets)

| Config | Layers | Clean | Sentiment flips (0°→180°) |
|---|---|---|---|
| max_sim_36_mid FULL | 48 | 120° | 1-4 / 150 (weak) |
| max_sim_36_mid_L31-41 | 11 | 360° | 1-3 / 150 (almost zero) |
| max_sim_36_mid_L26-46 | 21 | 360° | 1-3 / 150 |
| max_sim_36_mid_L20-47 | 28 | 360° | 1-4 / 150 |
| max_norm_33_post FULL | 48 | 45° (old metric) / ~135° (strict) | **27-30 / 150 (strong)** |
| max_norm_33_post_L28-38 | 11 | 360° | 0-4 / 150 |
| max_norm_33_post_L20-46 | 27 | 360° | 3-7 / 150 |

### The key diagnostic finding: the REP detector was over-flagging

My initial "flag rate" metric (3-gram repeats ≥4 times) counted
*successful enthusiastic outputs* as failures. For example, the 14B fp8
max_norm-FULL outputs at 300°-315° looked like:
> "How cool and classy! How amazing! How wonderful! #Amazing #Incredible #Legendary"

These are **successful sentiment-steered outputs** (positive
exaggeration) that triggered REP because of the `"How X!"` adjective
stack. A strict metric (trigram ≥6 repeats, OR single token dominates
≥35% of the output) correctly separates these from genuine pathology.

**Revised clean range for 14B fp8 max_norm FULL** under the strict metric:
~135° of usable data (30°-120° contiguous plus 180°-210°), with 225°-315°
as a real ~90° dead zone of actual REP loops. The analyzer should use
the strict metric going forward.

### Sentiment asymmetry on 14B

- **POS→NEG at 180° works cleanly** — 2% flagged, and the model
  reinterprets positive tweets as distress/empathy responses. E.g.
  "Phase 2 was a success" → "I know it's hard to stay positive when you're
  facing challenges, but you're not alone." This is coherent negative-
  valenced text and would count as a successful negative steer.
- **NEG→POS at 0° fails** — negative tweets produce hashtag soup:
  "pff, Life sucks sometimes!" → "#it'sawesome #indeed #nottobemore
  #thanmeresentment..." The model can't coherently generate positive
  text anchored to a negative input.

This asymmetry matches the instruction-tuning bias we observed on 7B/14B:
models handle "push positive toward negative" more gracefully than the
reverse.

### Recommended next experiments on H100

1. **Run both 14B and 32B unquantized** at `--angular-step 15`:
   ```bash
   bash setup_and_run.sh 14B
   bash setup_and_run.sh 32B
   ```
   (setup_and_run.sh defaults to `restate,echo_en,similar_tweet_en,rewrite`
   templates at 15° step.) Expected runtime on H100: 14B ~2h, 32B ~4h.

2. **Determine whether fp8 was the problem.** If 14B bf16 produces the
   same "no middle ground" pattern as fp8, then it's an intrinsic scale
   property, not a quantization artifact. If bf16 gives clean outputs
   with meaningful flips at the default config, report that FP8 should
   be avoided for angular steering.

3. **If 14B/32B still need layer-subset filtering**, use
   `filter_config_layers.py` with a middle range like L_peak ± 10. Peak
   layers: 14B max_sim=36, 14B max_norm=33; 32B auto-detects (check log).

4. **Update the analyzer's strict metric** (task carried over) so
   existing result CSVs get cleaner scores:
   - `analyze_template_discovery.py`: require 3-gram repeat ≥6 (not ≥4),
     add single-token dominance check (top token ≥35% of words).
   - This will retroactively show that earlier runs had lower real
     degeneration than reported.

## Open questions / future work

1. **Investigate the 120°–150° zone structurally** — measure activation norms
   and cosine similarity between rotated activations and the training
   distribution. This is the only failure pattern prompting hasn't fixed.
2. **Template ensemble** — route tweets through different templates depending
   on the target angle. Unexplored.
3. **Compare adaptive modes per template** — mode 0 beat mode 1 for `echo_en`
   but this isn't guaranteed for other templates.
4. **Re-extract 14B/32B sentiment directions from more than 80 prompts**
   if the direction turns out to be weak. The extraction script uses a
   fixed 100-statement hardcoded list per sentiment.

## Result CSVs in the repo

Under `results/`:

- `results_7B_20260412_15deg_restate_echo_en_similar_tweet_en_rewrite.csv.gz`
  — **Flagship 7B run at 15° granularity.** 24 angles × 4 templates + CAA +
  prompted, 457k rows gzipped (~55 MB). Uses `max_sim_19_mid` config
  (all layers).
- `results_3B_20260412_15deg_restate_echo_en_similar_tweet_en_rewrite.csv.gz`
  — **Flagship 3B run at 15° granularity.** Same structure as 7B, 457k rows
  gzipped (~51 MB). Uses the L20-30 filtered config
  (`max_sim_27_mid_L20-30-pca_0.npy`). `rewrite` is the best template on 3B
  (0.4% mean flagged, 360° clean at 30° granularity).
- `results_7B_20260411_restate_echo_en_similar_tweet_en_rewrite.csv.gz` —
  prior 30°-granularity 7B sweep (12 angles). Kept for completeness.
- `results_3B_20260411_restate_echo_en_similar_tweet_en_rewrite.csv.gz` —
  prior 30°-granularity 3B sweep.
- `results_32B_20260411_rewrite_en_similar_tweet_en.csv.gz` — 32B sweep
  with the two best templates from 32B discovery (rewrite_en, similar_tweet_en).
- `template_discovery_{7B,32B}_round*_mode0_scores.csv` — per-round template ×
  angle primary-flag summary statistics. These small score files justify
  the template winners.

Decompress any of the above with `gunzip -k <file>.gz`. The analyzer reads
`.csv.gz` directly, so you don't need to decompress for analysis.

**Intentionally not committed** (regeneratable or redundant):
- Raw template discovery CSVs (~33 MB each) — regenerate with
  `python run_template_discovery.py --model <key> --round N`
- `results_7B_20260411.csv` (the echo_en-only run, 39 MB) — redundant with
  the multi-template gzip above (same conditions, different stochastic outputs)
- Earlier 3B runs with the wrong configs (`max_norm_35_post` too weak,
  `max_sim_27_mid` full-layers destroyed outputs)
- Per-round log files
