# Reproduction Results

## Existing responses (original LFS direction, mode 4 = mode 0, 90°)

| Model | Substring (↓) | HarmBench (↑) | LlamaGuard (↑) |
|-------|---------------|---------------|-----------------|
| 3B paper | 0.1731 | 0.2019 | 0.2212 |
| 3B ours  | 0.1731 | 0.2115 | 0.2404 |
| 7B paper | 0.0577 | 0.3942 | 0.5288 |
| 7B ours  | 0.0577 | 0.3942 | 0.5385 |

HarmBench/LlamaGuard slight deviations due to 4-bit quantized classifiers.

## Fresh no-padding direction (our extraction), mode 0, 90°

| Model | Substring (↓) | HarmBench (↑) |
|-------|---------------|---------------|
| 3B | 0.0000 | 0.1635 |
| 7B | 0.1731 | 0.1346 |

## Fresh no-padding direction, mode 1, 90°

| Model | Substring (↓) | HarmBench (↑) | LlamaGuard (↑) |
|-------|---------------|---------------|-----------------|
| 3B | 0.0000 | 0.7596 | — |
| 7B | 0.0288 | 0.4904 | 0.6442 |

Mode 1 produces coherent, harmful responses that exceed paper HarmBench/LlamaGuard
targets but have near-zero refusal (direction is stronger than original).

## Key findings

- Angular steering at 90° with mode 0 (unconditional) on layernorm outputs reproduces
  the paper's "ablation" column
- No-padding extraction (one sample at a time via TL) recovers the correct best layer
  (25 for 3B, 19 for 7B)
- Fresh direction is qualitatively correct but quantitatively different from original
- Evaluation pipeline works: substring (exact), HarmBench (4-bit, ~validated),
  LlamaGuard (4-bit, ~validated)

## Files

- Existing responses: `output/Qwen2.5-{3,7}B-Instruct/harmful-en-dir_max_sim_*-pca_0-adaptive_4.json`
- Fresh 3B mode 0: `output/nopad_test/Qwen2.5-3B-Instruct/harmful-en-max_sim_25_mid-pca_0-rotated.json`
- Fresh 3B mode 1: `output/nopad_mode1_90.json`
- Fresh 7B mode 0: `output/nopad_7b_mode0_90.json`
- Fresh 7B mode 1: `output/nopad_7b_mode1_90.json`
- Steering configs: `output/nopad_*config/`
