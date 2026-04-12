#!/bin/bash
# Setup and run sentiment steering experiment on a fresh server.
#
# Usage:
#   bash setup_and_run.sh [MODEL_KEY] [TEMPLATES] [ANGULAR_STEP]
#
# Examples:
#   bash setup_and_run.sh 14B
#   bash setup_and_run.sh 14B restate,echo_en,similar_tweet_en,rewrite 15
#   bash setup_and_run.sh 32B rewrite_en,echo_en 15
#
# Defaults: TEMPLATES=restate,echo_en,similar_tweet_en,rewrite, ANGULAR_STEP=15
#
# Steps:
#   1. Install dependencies
#   2. Extract sentiment directions (both max_sim and max_norm) if missing
#   3. Run the full sentiment experiment
#
# Prerequisites: CUDA GPU with enough VRAM for the chosen model in bf16,
# git clone of this repo already done.
#
# H100-sized GPU notes:
#   - 14B bf16 needs ~28 GB. An H100 (80 GB) easily fits it.
#   - 32B bf16 needs ~64 GB. Still fits on H100.
#   - Use --angular-step 15 for fine-grained sweeps (24 angles, ~2x runtime).

set -e

MODEL_KEY="${1:-14B}"
TEMPLATES="${2:-restate,echo_en,similar_tweet_en,rewrite}"
ANGULAR_STEP="${3:-15}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Sentiment Steering Experiment Setup"
echo "  Model:         $MODEL_KEY"
echo "  Templates:     $TEMPLATES"
echo "  Angular step:  ${ANGULAR_STEP}° (=$((360 / ANGULAR_STEP)) angles)"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing dependencies ---"
pip install 'vllm>=0.11.0' numpy pandas tqdm scikit-learn transformers accelerate
# pytorch_pure extraction also needs these
pip install 'torch>=2.0.0' 'datasets>=2.14.0'

# ---------------------------------------------------------------------------
# 2. Extract sentiment directions if needed
# ---------------------------------------------------------------------------

# Map model keys to HuggingFace model IDs
declare -A MODEL_IDS=(
    ["3B"]="Qwen/Qwen2.5-3B-Instruct"
    ["7B"]="Qwen/Qwen2.5-7B-Instruct"
    ["14B"]="Qwen/Qwen2.5-14B-Instruct"
    ["32B"]="Qwen/Qwen2.5-32B-Instruct"
    ["Llama-3B"]="meta-llama/Llama-3.2-3B-Instruct"
    ["Llama-8B"]="meta-llama/Llama-3.1-8B-Instruct"
)

MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
MODEL_NAME="${MODEL_ID##*/}"
STMT_DIR="output/${MODEL_NAME}/STMT"

if [ -z "$MODEL_ID" ]; then
    echo "ERROR: Unknown model key '$MODEL_KEY'"
    echo "Available: ${!MODEL_IDS[@]}"
    exit 1
fi

# Check if STMT config already exists
if ls "$STMT_DIR"/steering_config-*.npy 1>/dev/null 2>&1; then
    echo ""
    echo "--- Sentiment config already exists for $MODEL_NAME ---"
    ls "$STMT_DIR"/steering_config-*.npy
else
    echo ""
    echo "--- Extracting sentiment directions for $MODEL_NAME ---"
    echo "This may take a while (forward passes through the model)..."
    cd pytorch_pure
    python extract_directions_sentiment.py \
        --model "$MODEL_ID" \
        --output-dir ../output \
        --language en \
        --n-samples 512 \
        --batch-size 8 \
        --positions mid post \
        --strategy both
    cd "$SCRIPT_DIR"
    echo "Extraction complete. Configs:"
    ls "$STMT_DIR"/steering_config-*.npy
fi

# ---------------------------------------------------------------------------
# 3. Run the experiment
# ---------------------------------------------------------------------------
echo ""
echo "--- Running experiment ---"
python run_sentiment_experiment.py \
    --model "$MODEL_KEY" \
    --templates "$TEMPLATES" \
    --angular-step "$ANGULAR_STEP" \
    --output-dir results/

echo ""
echo "============================================"
echo "  Done! Results saved to results/"
echo "============================================"
