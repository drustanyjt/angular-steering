#!/bin/bash
# Run directional ablation experiment for Qwen2.5-3B-Instruct
# Replicates the Ablation column from the paper table

set -e
cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR="./output"
LANGUAGE="en"

echo "=== Step 1: Extract steering directions (TransformerLens, matches notebook) ==="
conda run -n MA4198 python extract_directions_tl.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --language "$LANGUAGE" \
    --n-samples 512 \
    --batch-size 16

echo ""
echo "=== Step 2a: All-layer ablation + substring matching ==="
conda run -n MA4198 python pytorch_pure/run_ablation.py \
    --model "$MODEL" \
    --config-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --language "$LANGUAGE" \
    --batch-size 8 \
    --max-tokens 512 \
    --strategy max_sim \
    --eval-methods substring_matching

echo ""
echo "=== Step 2b: Single-layer ablation + substring matching ==="
conda run -n MA4198 python pytorch_pure/run_ablation.py \
    --model "$MODEL" \
    --config-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --language "$LANGUAGE" \
    --batch-size 8 \
    --max-tokens 512 \
    --strategy max_sim \
    --single-layer \
    --eval-methods substring_matching

echo ""
echo "=== Step 3: HarmBench evaluation (separate process to avoid CUDA conflict) ==="
conda run -n MA4198 python -c "
import sys, json, numpy as np
sys.path.insert(0, '.')
from llm_activation_control.utils import get_harmful_instructions
from evaluate_jailbreak import evaluate_jailbreak

_, data_test = get_harmful_instructions()

output = 'output/Qwen2.5-3B-Instruct'
for suffix in ['all_layers', 'single_layer']:
    path = f'{output}/harmful-en-ablation-{suffix}.json'
    try:
        with open(path) as f:
            responses = json.load(f)
        scores = evaluate_jailbreak(data_test, responses, methodologies=['harmbench'])
        asr = float(np.mean(scores))
        print(f'HarmBench {suffix}: harmful_rate(paper ↑)={asr:.4f}')
        # save alongside existing eval file
        eval_path = f'{output}/eval-ablation-{suffix}-en.json'
        with open(eval_path) as f:
            ev = json.load(f)
        ev['harmbench'] = {'baseline_paper': 0.0, 'ablation_paper': asr,
                           'baseline_asr': 0.0, 'ablation_asr': asr}
        with open(eval_path, 'w') as f:
            json.dump(ev, f, indent=4)
        print(f'  Updated {eval_path}')
    except FileNotFoundError:
        print(f'Skipping {suffix}: file not found')
"

echo ""
echo "=== Done ==="
