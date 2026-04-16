# Tweet Sentiment Angular Steering (TSAD)

Reproduction guide for the angular-steering experiments reported in the CS4248
final report (Part II: Activation Steering).

## Branch

All experiments in the report were run from the **`drus`** branch of this repo
(<https://github.com/drustanyjt/angular-steering>).

```bash
git clone https://github.com/drustanyjt/angular-steering.git
cd angular-steering
git checkout drus
```

The `ruiheng` branch contains teammate work on classical and neural sentiment
classifiers (Part I of the report) and is independent of the steering pipeline.

## Environment

```bash
pip install -r drus_requirements.txt
```

A single consumer-grade discrete GPU is sufficient for the 3B and 7B sweeps. The
14B and 32B sweeps were offloaded to a shared workstation GPU.

## Data

Place the Tweet Sentiment Analysis Dataset (TSAD) at `tsad/test.csv` and
`tsad/train.csv`. The CSVs must expose the columns `text` and `sentiment`
(`positive` / `neutral` / `negative`). The raw dataset is available on Kaggle
and is not committed to this repo.

## Pipeline

Three stages. All commands assume the repo root as the working directory.

### 1. Extract steering directions

```bash
python pytorch_pure/extract_directions_tsad.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir output/Qwen2.5-7B-Instruct \
    --n-samples 512 \
    --positions mid post \
    --strategy both
```

This runs a CAA-style **pairwise** difference extraction using a curated set
of hand-written, position-matched positive and negative emotional statements
(see `get_positive_instructions` / `get_negative_instructions` in
`pytorch_pure/utils.py`). For each layer, it computes the mean of per-pair
activation differences, then takes the first principal component of the
per-layer candidate directions (following Vu & Nguyen 2025) as the second
axis. The result is a hybrid steering plane: CAA-style pairwise first axis,
PCA-over-layers second axis. Output is written to
`output/<model>/STMT/steering_config-*.npy`.

Re-run with `--model Qwen/Qwen2.5-{3B,14B,32B}-Instruct` for the other sizes
reported in the paper.

### 2. Generate steered responses across the angle sweep

The entry point is `steer_tsad_tweets.py` (vLLM-backed). Sweep
$\theta \in \{0°, 15°, \ldots, 345°\}$ across the seven prompt templates
described in the report. The `mainpy.py` / `mainpy_7b.py` wrappers run the full
sweep end-to-end; see their docstrings for per-template invocations. Outputs
are written as CSVs in `ignore_data/` with per-row sentiment classifier
probabilities (`prob_positive`, `prob_neutral`, `prob_negative`) attached.

### 3. Generate the polar plots (used in the report)

```bash
python plot_sentiment_polar.py \
    --csv ignore_data/results_7B_20260412_15deg_7templates_full_pred.csv \
    --out figures/sentiment_polar_7B
```

Swap the `--csv` / `--out` arguments for the other model sizes. This produces
one polar PNG per prompt template plus a `neutral_all_templates.png` overlay.
These are the figures reproduced in Appendix A of the report.

## Minimal sanity check

To sanity-check the pipeline end-to-end on a single template and angle without
a full sweep:

```bash
python steer_tsad_tweets.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --config output/Qwen2.5-7B-Instruct/directions.npy \
    --csv tsad/test.csv \
    --sentiment positive \
    --n 5 \
    --target_degree 180.0
```

This should produce five tweets steered toward the negative pole, printed to
stdout.
