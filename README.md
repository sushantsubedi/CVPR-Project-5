# ACT + KAT-style In-Context Imitation (Transfer Cube)


---

## How to use this repo

### 0. Prerequisites

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

### 3. Run Step 5 (seen + unseen + ablations + aggregation)

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
├── results_main_seen_unseen.csv      main seen/unseen on demos_5 (K=10, M=20)
├── results_ablation_ndemos_{10,20}_unseen.csv
├── results_ablation_K{5,10,20}_unseen.csv
├── results_ablation_M{10,20,40}_unseen.csv
├── results_ablation_model_{gemma4_26b,llama3_2_latest}_unseen.csv
├── evaluation_summary.json           aggregated summary
└── evaluation_tables.md              Markdown table for the report
```

Or open [`KAT_ICL.ipynb`](KAT_ICL.ipynb) and run all cells — it shells out to the same scripts inside the activated venv and shows the sanity-check artifacts (keypoint overlay PNG, gt-replay video, ICL rollout video) inline.

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


---

## Repo layout

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
└── evaluate.py                   evaluation entry: run / sweep / aggregate

reports/KAT_ICL_Report.md         full write-up
KAT_ICL.ipynb                     single runnable notebook (Steps 1–5)
data/transfer_cube/DATA_MANIFEST.md   dataset & split contract
requirements-kat.txt              unified deps (ACT sim + KAT-ICL)
```

---

## Seen vs. unseen (Step 5 requirement)

* `subsets/demos_{5,10,20}` hold episodes **0..N-1** of the seed-0 run — used as in-context demonstrations.
* `subsets/test_5` holds episodes **20..24** — **never present** in any demo folder.
* `scripts/evaluate.py` tags every result row with `split ∈ {seen, unseen}`:
  * **seen** = leave-one-out within the demo folder (query is one demo, prompt has the other N-1).
  * **unseen** = queries are episodes 20..24 with fresh box poses from the same seeded run.

## Ablations covered

| Ablation | Values | Holds fixed |
|----------|--------|-------------|
| #demos | demos_5 (main) / demos_10 / demos_20 | K=10, M=20, model=gemma4:26b, unseen |
| K keypoints | 5 / 10 / 20 | demos_5, M=20, model=gemma4:26b, unseen |
| M action tokens | 10 / 20 / 40 | demos_5, K=10, model=gemma4:26b, unseen |
| LLM | `gemma4:26b` vs `llama3.2:latest` | demos_5, K=10, M=20, unseen |

---

### ACT base files
- `imitate_episodes.py` — train + evaluate ACT
- `policy.py` — ACT policy adaptor
- `detr/` — model definitions (modified DETR)
- `sim_env.py`, `ee_sim_env.py` — MuJoCo + DM_Control envs
- `scripted_policy.py` — scripted oracle policies
- `record_sim_episodes.py`, `visualize_episodes.py`
- `constants.py`, `utils.py`
