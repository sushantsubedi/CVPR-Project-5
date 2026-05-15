# ACT + Project 5 (KAT-style In-Context Imitation)

This repo combines:

1. **ACT** (Action Chunking with Transformers) — the original sim + training code, unchanged.
2. **Project 5 (P5)** — a KAT-style in-context imitation pipeline on top, for the **Transfer Cube** task.

> P5 pipeline: top RGB-D → DINO anchored keypoints (`KP i x y d`) + quantized 14-D waypoints (`WP[i] L LG R RG`) → Ollama few-shot ICL with JSON schema → decode + MuJoCo replay → success/reward CSV.
>
> **Scope.** P5 Steps 1–5; optional fine-tuning (Step 6) is **not** implemented.

---

## How to use this repo

### 0. Prerequisites

* macOS or Linux, **Python ≥ 3.9**
* [`uv`](https://github.com/astral-sh/uv) (`brew install uv` or `pipx install uv`) — the included `./.venv` is a uv-managed virtual environment
* A running [Ollama](https://ollama.ai) server with at least one of: `gemma4:26b`, `llama3.2:latest`
  ```bash
  ollama serve            # in one terminal
  ollama pull gemma4:26b
  ollama pull llama3.2:latest
  ```

### 1. One-time setup

```bash
# create / refresh the uv venv and install ALL dependencies (ACT sim + KAT-ICL)
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements-kat.txt    # single consolidated requirements file
cd detr && pip install -e . && cd ..      # ACT model package (editable)

# DINO ViT-S/16 weights (~83 MB, downloads to assets/weights/)
python3 scripts/download_dino_weights.py
```

> Conda alternative: `conda env create -f conda_env.yaml && conda activate aloha` still works for the ACT base code. The unified `requirements-kat.txt` is recommended for P5.

### 2. Collect data once (fixed seed)

```bash
source .venv/bin/activate
python3 record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir data/transfer_cube/base \
  --num_episodes 25 --downsample_rate 10 --seed 0
python3 scripts/make_subsets.py --base_dir data/transfer_cube/base
```

This produces `data/transfer_cube/{base, subsets/demos_5, subsets/demos_10, subsets/demos_20, subsets/test_5}`.

### 3. Run P5 Step 5 (seen + unseen + ablations + aggregation)

One command:

```bash
python3 scripts/evaluate.py sweep --collection_seed 0 \
  --models 'gemma4:26b,llama3.2:latest'
```

Output appears under `artifacts/`:

```
artifacts/
├── prompts/<run_id>.txt              full LLM prompt
├── responses/<run_id>.json           parsed JSON response
├── replays/<run_id>.mp4              rollout video
├── results_p5_main_seen_unseen.csv   main seen/unseen on demos_20
├── results_ablation_ndemos_{5,10,20}_unseen.csv
├── results_ablation_K{5,10,20}_unseen.csv
├── results_ablation_M{10,20,40}_unseen.csv
├── results_ablation_model_{gemma4_26b,llama3_2_latest}_unseen.csv
├── p5_step5_summary.json             aggregated summary
└── p5_step5_tables.md                Markdown table for the report
```

Or open [`P5_KAT_ICL_Complete.ipynb`](P5_KAT_ICL_Complete.ipynb) and run all cells — it shells out to the same scripts inside the activated venv.

### 4. (Optional) Inspect individual pieces

```bash
# Print OBS + ACT tokens for one episode (+ KP overlay PNG)
python3 scripts/show_tokens.py \
  --episode_hdf5 data/transfer_cube/subsets/demos_5/episode_0.hdf5 \
  --K 10 --M 20 --overlay artifacts/keypoints_overlay.png

# Decode-free upper bound: replay the stored /action arrays
python3 scripts/gt_replay.py --dataset_dir data/transfer_cube/subsets/test_5

# Replay a saved LLM response on one test episode
python3 scripts/replay_response.py \
  --response artifacts/responses/<run_id>.json \
  --test_episode_hdf5 data/transfer_cube/subsets/test_5/episode_20.hdf5 \
  --demos_dir data/transfer_cube/subsets/demos_5 --M 20

# Aggregate results CSVs only (no LLM calls)
python3 scripts/evaluate.py aggregate --glob 'artifacts/results_*.csv'
```

> **uv tip:** instead of `source .venv/bin/activate && python3 …` you can prefix every command with `uv run --python .venv/bin/python3 …`; it will pick up the same interpreter.

---

## Repo layout (P5 parts)

```
act_kat/                          KAT library (8 modules)
├── episodes.py                   HDF5 helpers
├── vision_tokens.py              DINO ViT-S/16 + FPS + patch->pixel
├── keypoint_anchors.py           paper-style anchored keypoints + tokenization
├── action_tokens.py              14-D action quantizer + TRAJ/WP formatter
├── ollama_client.py              Ollama HTTP + artifact logging
├── replay.py                     sim replay + linear/hold upsampling
└── icl.py                        prompt build / generate / parse / replay

scripts/                          6 CLI entry points
├── download_dino_weights.py
├── make_subsets.py
├── show_tokens.py
├── replay_response.py
├── gt_replay.py
└── evaluate.py                   Step 5 entry: run / sweep / aggregate

reports/P5_KAT_ICL_Report.md      full write-up
P5_KAT_ICL_Complete.ipynb         single runnable notebook
data/transfer_cube/DATA_MANIFEST.md   dataset & split contract
requirements-kat.txt              unified deps (ACT sim + KAT-ICL)
```

---

## Seen vs. unseen (P5 Step 5 requirement)

* `subsets/demos_{5,10,20}` hold episodes **0..N-1** of the seed-0 run — used as in-context demonstrations.
* `subsets/test_5` holds episodes **20..24** — **never present** in any demo folder.
* `scripts/evaluate.py` tags every result row with `split ∈ {seen, unseen}`:
  * **seen** = leave-one-out within the demo folder (query is one demo, prompt has the other N-1).
  * **unseen** = queries are episodes 20..24 with fresh box poses from the same seeded run.

## Ablations covered

| Ablation | Values | Holds fixed |
|----------|--------|-------------|
| #demos | demos_5 / demos_10 / demos_20 | K=10, M=20, model=gemma4:26b, unseen |
| K keypoints | 5 / 10 / 20 | demos_5, M=20, model=gemma4:26b, unseen |
| M action tokens | 10 / 20 / 40 | demos_5, K=10, model=gemma4:26b, unseen |
| LLM | `gemma4:26b` vs `llama3.2:latest` | demos_5, K=10, M=20, unseen |

---

## Original ACT (training, evaluation)

`imitate_episodes.py` and friends are untouched and still work as in the upstream repo:

```bash
# Train ACT on transfer cube
python3 imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt dir> \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

# Evaluate (same command + --eval)
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt dir> --policy_class ACT --eval
```

Typical success after full training: ~90% on transfer cube, ~50% on insertion. See [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

### ACT base files
- `imitate_episodes.py` — train + evaluate ACT
- `policy.py` — ACT policy adaptor
- `detr/` — model definitions (modified DETR)
- `sim_env.py`, `ee_sim_env.py` — MuJoCo + DM_Control envs
- `scripted_policy.py` — scripted oracle policies
- `record_sim_episodes.py`, `visualize_episodes.py`
- `constants.py`, `utils.py`
