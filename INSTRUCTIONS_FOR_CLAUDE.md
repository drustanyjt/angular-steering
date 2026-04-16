# Instructions for Claude — Random Direction Control Experiment

## Context

This is Drustan's MA4198 (NUS Maths capstone) project on refusal steering in LLMs.
The repo is a fork of Ian Hieu's `angular-steering` codebase. We are running a
**control experiment**: angular steering with a random direction instead of the
learned refusal direction, to confirm the refusal direction is specifically
responsible for jailbreak success (not just any activation perturbation).

- **Repo:** `https://github.com/drustanyjt/angular-steering.git`, branch `drus`
- **Conda env:** `MA4198`
- **Working dir:** wherever the repo is cloned on this server

---

## Step 1 — Run the pipeline (steps 1–3 automated)

```bash
bash run_random_direction.sh
```

This runs for both `Qwen2.5-3B-Instruct` and `Qwen2.5-7B-Instruct`:
1. Extract steering directions via TransformerLens (skips if `steering_config-en-dir_max_sim_*-pca_0.npy` already exists in `output/<model>/`)
2. Generate random direction configs (`generate_random_config.py` — replaces the PCA plane direction with a seeded Gaussian unit vector)
3. Sweep all 36 angles (0°–350°, 10° steps) and save responses to `output/<model>/harmful-en-dir_max_sim_*-dir_random-adaptive_1.json`

Single model only:
```bash
bash run_random_direction.sh --model Qwen/Qwen2.5-3B-Instruct
```

---

## Step 2 — Evaluate

`evaluate_jailbreak.py` is hardcoded rather than CLI-driven, so call it via Python inline.

**Qwen2.5-3B-Instruct:**
```bash
conda run -n MA4198 python - <<'EOF'
from evaluate_jailbreak import evaluate_model
from configs import MAX_SIM_DIR_ID

model_id = "Qwen/Qwen2.5-3B-Instruct"
dir_id   = MAX_SIM_DIR_ID[model_id]   # "dir_max_sim_25_mid"

for method in ["substring_matching", "harmbench", "llamaguard3"]:
    evaluate_model(
        model_id=model_id,
        method=method,
        data_type="harmful",
        language="en",
        output_path="output/",
        included_direction_ids=[dir_id, "dir_random"],
        adaptive_mode=1,
    )
EOF
```

**Qwen2.5-7B-Instruct:**
```bash
conda run -n MA4198 python - <<'EOF'
from evaluate_jailbreak import evaluate_model
from configs import MAX_SIM_DIR_ID

model_id = "Qwen/Qwen2.5-7B-Instruct"
dir_id   = MAX_SIM_DIR_ID[model_id]   # "max_sim_19_mid"

for method in ["substring_matching", "harmbench", "llamaguard3"]:
    evaluate_model(
        model_id=model_id,
        method=method,
        data_type="harmful",
        language="en",
        output_path="output/",
        included_direction_ids=[dir_id, "dir_random"],
        adaptive_mode=1,
    )
EOF
```

> `harmbench` loads a 13B classifier; `llamaguard3` uses vLLM. Both need a GPU.
> Run `substring_matching` first as a quick sanity check — it is CPU-friendly.

---

## Step 3 — Push results

```bash
git add output/Qwen2.5-3B-Instruct/harmful-en-*dir_random*.json
git add output/Qwen2.5-3B-Instruct/eval-mode_1-*-harmful-en.json
git add output/Qwen2.5-7B-Instruct/harmful-en-*dir_random*.json
git add output/Qwen2.5-7B-Instruct/eval-mode_1-*-harmful-en.json
git commit -m "Add random direction control experiment results (3B + 7B)"
git push origin drus
```

---

## What to expect

Results land in `output/<model>/eval-mode_1-{method}-harmful-en.json` under the
key `dir_max_sim_*-dir_random`. Based on Ian's prior run, expect the random
direction to achieve moderate jailbreak rates (HarmBench ~0.13–0.35,
LlamaGuard3 ~0.17–0.41) vs. the refusal direction peak (~0.90–0.97),
confirming specificity of the learned direction.

---

*This file should be deleted after ingesting.*
