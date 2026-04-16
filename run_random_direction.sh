#!/bin/bash
# Run angular steering experiment with a RANDOM direction (control experiment).
#
# Purpose: verify that the structured refusal direction drives jailbreak success,
# not just any activation perturbation. Sweeps all angles with a random plane
# (refusal direction replaced by Gaussian noise) and evaluates jailbreak rate.
#
# Pipeline:
#   1. Extract steering directions via TransformerLens (skip if pca_0 config exists)
#   2. Generate random direction configs from the pca_0 configs
#   3. Sweep all angles (10° steps) and generate responses with random direction
#   4. Evaluate with substring matching (+ optionally HarmBench / LlamaGuard3)
#
# Usage:
#   bash run_random_direction.sh                          # both 3B and 7B, 10° steps
#   bash run_random_direction.sh --model Qwen/Qwen2.5-3B-Instruct
#   bash run_random_direction.sh --angle-step 30          # quick 12-angle test

set -e
cd "$(dirname "$0")"

MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
OUTPUT_DIR="./output"
LANGUAGE="en"
N_SAMPLES=512
BATCH_SIZE=8
MAX_TOKENS=512
ANGLE_STEP=10
ADAPTIVE_MODE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODELS=("$2"); shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --angle-step)  ANGLE_STEP="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Random Direction Control Experiment"
echo "========================================"
echo "  Models:      ${MODELS[*]}"
echo "  Angle step:  ${ANGLE_STEP}°"
echo "  Batch size:  ${BATCH_SIZE}"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    MODEL_OUT="$OUTPUT_DIR/$MODEL_NAME"
    echo "----------------------------------------"
    echo "Model: $MODEL_NAME"
    echo "----------------------------------------"

    # ── Step 1: extract directions if pca_0 config is missing ────────────────
    PCA_GLOB=$(ls "$MODEL_OUT"/steering_config-en-dir_max_sim_*-pca_0.npy 2>/dev/null | head -1)
    if [[ -z "$PCA_GLOB" ]]; then
        echo "Step 1: pca_0 config not found — extracting directions..."
        conda run -n MA4198 python extract_directions_tl.py \
            --model "$MODEL" \
            --output-dir "$OUTPUT_DIR" \
            --language "$LANGUAGE" \
            --n-samples "$N_SAMPLES" \
            --batch-size "$BATCH_SIZE"
    else
        echo "Step 1: pca_0 config found ($PCA_GLOB) — skipping extraction"
    fi

    # ── Step 2: generate random direction configs ─────────────────────────────
    RANDOM_GLOB=$(ls "$MODEL_OUT"/steering_config-en-dir_max_sim_*-dir_random.npy 2>/dev/null | head -1)
    if [[ -z "$RANDOM_GLOB" ]]; then
        echo "Step 2: generating random direction configs..."
        conda run -n MA4198 python generate_random_config.py
    else
        echo "Step 2: random config found ($RANDOM_GLOB) — skipping"
    fi

    # ── Step 3: generate responses for the random direction ───────────────────
    RESPONSE_GLOB=$(ls "$MODEL_OUT"/harmful-en-dir_max_sim_*-dir_random-adaptive_${ADAPTIVE_MODE}.json 2>/dev/null | head -1)
    if [[ -z "$RESPONSE_GLOB" ]]; then
        echo "Step 3: generating responses (${ANGLE_STEP}° steps)..."
        conda run -n MA4198 python pytorch_pure/generate_responses.py \
            --model "$MODEL" \
            --config-dir "$OUTPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --language "$LANGUAGE" \
            --batch-size "$BATCH_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            --angle-step "$ANGLE_STEP" \
            --adaptive-mode "$ADAPTIVE_MODE" \
            --strategy-filter "dir_random"
    else
        echo "Step 3: response file found ($RESPONSE_GLOB) — skipping"
    fi

    echo ""
    echo "Step 4: evaluation"
    echo "  evaluate_jailbreak.py is hardcoded — run the snippet below:"
    echo ""
    echo "  conda run -n MA4198 python - <<'EOF'"
    echo "  from evaluate_jailbreak import evaluate_model"
    echo "  from configs import MAX_SIM_DIR_ID"
    echo "  model_id = \"$MODEL\""
    echo "  dir_id   = MAX_SIM_DIR_ID[model_id]"
    echo "  for method in [\"substring_matching\", \"harmbench\", \"llamaguard3\"]:"
    echo "      evaluate_model("
    echo "          model_id=model_id,"
    echo "          method=method,"
    echo "          data_type=\"harmful\","
    echo "          language=\"en\","
    echo "          output_path=\"output/\","
    echo "          included_direction_ids=[dir_id, \"dir_random\"],"
    echo "          adaptive_mode=1,"
    echo "      )"
    echo "  EOF"
    echo ""
done

echo "========================================"
echo "Response generation done."
echo "Run the evaluation snippet above for each model."
echo "========================================"
