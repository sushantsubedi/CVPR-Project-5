# Project 5 Report — KAT-style In-Context Imitation Learning (ACT Transfer Cube)

**Course:** Computer Vision and Pattern Recognition
**Project:** Fine-Tuning LLMs for Keypoint Action Token Imitation Learning in Simulation (P5)
**Reference:** Di Palo & Johns, *Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics* (RSS 2024)

**Scope.** P5 Steps 1–5 on the **Transfer Cube** task only. Optional fine-tuning (Step 6) is **not performed** (out of scope per project brief).

---

## 0. How to use this repository

This report sits next to a working codebase; the same pipeline can be reproduced in a few commands.

### 0.1 Prerequisites

* macOS or Linux, **Python ≥ 3.9**
* [`uv`](https://github.com/astral-sh/uv) — the included `./.venv` is a uv-managed virtual environment
* A running [Ollama](https://ollama.ai) server with the models used here:
  ```bash
  ollama serve
  ollama pull gemma4:26b
  ollama pull llama3.2:latest
  ```

### 0.2 One-time install

```bash
uv venv .venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements-kat.txt   # single consolidated requirements file
cd detr && pip install -e . && cd ..     # ACT model package (editable)
python3 scripts/download_dino_weights.py # DINO ViT-S/16 weights, ~83 MB
```

### 0.3 Reproduce all P5 results

```bash
source .venv/bin/activate

# Step 2 — data (fixed seed → deterministic seen/unseen split)
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted \
  --dataset_dir data/transfer_cube/base --num_episodes 25 \
  --downsample_rate 10 --seed 0
python3 scripts/make_subsets.py --base_dir data/transfer_cube/base

# Steps 3–5 — tokens, ICL, eval, four ablations, aggregation (one command)
python3 scripts/evaluate.py sweep --collection_seed 0 \
  --models 'gemma4:26b,llama3.2:latest'
```

Equivalent: open `P5_KAT_ICL_Complete.ipynb` and run all cells.

### 0.4 What you get under `artifacts/`

| File | Meaning |
|------|---------|
| `prompts/<id>.txt`, `responses/<id>.json` | LLM I/O per query |
| `replays/<id>.mp4` | rollout video per query |
| `results_p5_main_seen_unseen.csv` | seen + unseen on `demos_20` |
| `results_ablation_ndemos_{5,10,20}_unseen.csv` | #demos ablation |
| `results_ablation_K{5,10,20}_unseen.csv` | keypoint count ablation |
| `results_ablation_M{10,20,40}_unseen.csv` | action waypoint count ablation |
| `results_ablation_model_{gemma4_26b,llama3_2_latest}_unseen.csv` | LLM ablation |
| `p5_step5_summary.json`, `p5_step5_tables.md` | aggregated tables |

Every CSV row carries `split, model, K, M` so any cross-tab is one `pandas` away.

---

## 1. Abstract

We implement a KAT-style pipeline that maps top-view RGB-D observations and 14-D bimanual joint trajectories into discrete text tokens, prompts a local large language model (Ollama) for in-context imitation, parses the JSON response, and replays it in MuJoCo. Observation tokens use **DINO ViT-S/16 anchored keypoints** (paper-style best-buddy correspondences) with depth; action tokens use **uniform per-dimension quantization** of M waypoints. We evaluate few-shot ICL with explicit **seen / unseen** splits and run **four ablations** (number of demos, K keypoints, M waypoints, and LLM model). The whole project is reproducible from one Jupyter notebook or one shell sequence.

---

## 2. Method (short)

### 2.1 Observation tokens — vision-only

1. **Anchor selection (once per demo folder):** extract DINO patch descriptors for two frames of `episode_0` (t=0 and t=T/2). Take **best-buddy** correspondences, then run FPS on the buddy set to keep K anchors.
2. **Localize per frame:** each anchor descriptor is matched to the patch grid by cosine NN → stable index `i` across demos and queries.
3. **Serialize:**

```
OBS_START
KP 0 x=2  y=2  d=15
KP 1 x=14 y=8  d=12
...
OBS_END
```

`x,y` are normalized image coordinates quantized to `obs_bins` (default 16). `d` is depth clipped to [0, 2] m and quantized. No simulator ground truth is used.

### 2.2 Action tokens

* `M` waypoints uniformly subsampled over an episode.
* 14-D per-dimension uniform quantizer fit over the demo folder (default `bins=32`).
* Demo block mirrors the JSON schema exactly:

```
TRAJ_START
WP[i=0] L=[47,25,45,12,27,0] LG=0 R=[19,21,61,40,0,22] RG=0
...
TRAJ_END
```

LG/RG ∈ {0, 1} (0 = open, 1 = closed). The LLM is constrained to return JSON `{"actions":[{"L":[…],"LG":0,"R":[…],"RG":0}, …]}` with exactly M items.

### 2.3 ICL → sim

* Few-shot prompt = demo OBS+TRAJ pairs, then the query OBS, then a JSON instruction (`act_kat/icl.py::build_prompt`).
* Ollama `/api/generate` with JSON schema and `temperature=0`.
* Decode codes, **linearly upsample** M waypoints to 400 steps, set `BOX_POSE` from the query HDF5's `env_state0`, replay with `act_kat/replay.py`.

---

## 3. Dataset and splits (Step 5.1 — fixed seed, seen / unseen)

```
data/transfer_cube/
├── DATA_MANIFEST.md
├── base/                       # 25 episodes recorded with --seed 0
│   └── episode_{0..24}.hdf5    # attrs: collection_seed, episode_index, env_state0
└── subsets/
    ├── demos_5/                # 0..4
    ├── demos_10/               # 0..9
    ├── demos_20/               # 0..19
    └── test_5/                 # 20..24  (held out — never in any demos_* folder)
```

| Split | Definition | Query episodes (for `demos_20` setup) | # queries |
|-------|------------|---------------------------------------|----------:|
| **Seen** | Leave-one-out on the demo folder: query is one demo, prompt holds the other 19. | 0–19 | 20 |
| **Unseen / shifted** | Held-out test folder; box poses still come from the same seeded run. | 20–24 | 5 |

The `evaluate.py` script tags every CSV row with `split ∈ {seen, unseen}` and computes both rates in the summary JSON.

---

## 4. Evaluation, ablations and reproducibility (Step 5)

### 4.1 One-command sweep

```bash
source .venv/bin/activate
python3 scripts/evaluate.py sweep --collection_seed 0
```

Produces, under `artifacts/`:

* `results_p5_main_seen_unseen.csv` — main seen + unseen on `demos_20` (K=10, M=20).
* `results_ablation_ndemos_{5,10,20}_unseen.csv` — **#demos ablation**.
* `results_ablation_K{5,10,20}_unseen.csv` — **keypoint count ablation**.
* `results_ablation_M{10,20,40}_unseen.csv` — **action-token count ablation**.
* `results_ablation_model_{gemma4_26b,llama3_2_latest}_unseen.csv` — **LLM model ablation**.
* `p5_step5_summary.json`, `p5_step5_tables.md` — aggregated table.

Each CSV row: `split, episode_id, model, K, M, n_demos_in_prompt, demo_ids_in_prompt, success, max_reward, parse_ok, parse_errors`.

### 4.2 Single configuration

```bash
python3 scripts/evaluate.py run \
  --demos_dir data/transfer_cube/subsets/demos_5 \
  --tests_dir data/transfer_cube/subsets/test_5 \
  --eval_split both --K 10 --M 20
```

### 4.3 Ablations (required: at least one — we run four)

| Ablation | Values | Holds fixed |
|----------|--------|-------------|
| **#demos** | demos_5 / demos_10 / demos_20 | K=10, M=20, model=gemma4:26b, unseen |
| **K keypoints** | 5 / 10 / 20 | demos_5, M=20, model=gemma4:26b, unseen |
| **M action tokens** | 10 / 20 / 40 | demos_5, K=10, model=gemma4:26b, unseen |
| **Model (LLM)** | `gemma4:26b` (mid) vs `llama3.2:latest` (small) | demos_5, K=10, M=20, unseen |

Rows go into separate CSVs above; every row also records the `model`, `K`, `M` columns so cross-tabulation stays trivial. The aggregator emits one Markdown table.

The model ablation isolates the effect of **LLM capability** at constant token budget: same K=10 OBS tokens, same M=20 action waypoints, same five test episodes. `llama3.2` is ~3B parameters vs `gemma4:26b`'s ~26B parameters — a ~10× scale difference under identical prompt / schema constraints.

### 4.4 Results table

After the sweep finishes:

```bash
cat artifacts/p5_step5_tables.md
```

| CSV | N | Seen SR | Unseen SR | Overall SR | Parse OK |
|-----|--:|--------:|----------:|-----------:|---------:|
| `results_p5_main_seen_unseen.csv` | 25 | … | … | — | … |
| `results_ablation_ndemos_5_unseen.csv` | 5 | — | … | — | … |
| `results_ablation_ndemos_10_unseen.csv` | 5 | — | … | — | … |
| `results_ablation_ndemos_20_unseen.csv` | 5 | — | … | — | … |
| `results_ablation_K{5,10,20}_unseen.csv` | 5 each | — | … | — | … |
| `results_ablation_M{10,20,40}_unseen.csv` | 5 each | — | … | — | … |
| `results_ablation_model_gemma4_26b_unseen.csv` | 5 | — | … | — | … |
| `results_ablation_model_llama3_2_latest_unseen.csv` | 5 | — | … | — | … |

(Values are filled by `evaluate.py aggregate` from your run.)

### 4.5 Reproducibility checklist (P5 Step 5.3 / 5.4)

| Item | Provided |
|------|----------|
| Clean README | `README.md` |
| Environment file | `requirements-kat.txt` and base ACT `conda_env.yaml` |
| Data manifest | `data/transfer_cube/DATA_MANIFEST.md` |
| Evaluation script | `scripts/evaluate.py` |
| Results CSV | `artifacts/results_*.csv` (generated by sweep) |
| One-command reproduction | `python3 scripts/evaluate.py sweep --collection_seed 0` (after data + DINO weights) |

Four-command full re-run from a uv-activated venv:

```bash
source .venv/bin/activate       # uv-managed venv (see README)
python3 scripts/download_dino_weights.py
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted \
  --dataset_dir data/transfer_cube/base --num_episodes 25 --downsample_rate 10 --seed 0
python3 scripts/make_subsets.py --base_dir data/transfer_cube/base
python3 scripts/evaluate.py sweep --collection_seed 0
```

Or run `P5_KAT_ICL_Complete.ipynb` top to bottom.

---

## 5. Sanity checks (Step 3)

| Check | Command |
|-------|---------|
| Print OBS + ACT tokens + KP overlay PNG | `python3 scripts/show_tokens.py --episode_hdf5 data/.../episode_0.hdf5 --K 10 --M 20 --overlay artifacts/kp.png` |
| Decode-free GT replay (upper bound) | `python3 scripts/gt_replay.py --dataset_dir data/transfer_cube/subsets/demos_5` |
| Replay any saved LLM response | `python3 scripts/replay_response.py --response artifacts/responses/<id>.json --test_episode_hdf5 data/.../episode_20.hdf5 --demos_dir data/transfer_cube/subsets/demos_5 --M 20` |

---

## 6. Failure analysis (observed)

1. **Degenerate output** — the LLM occasionally emits identical waypoints (`L`, `R` repeated) → arms freeze; `parse_ok=1` but `success=0`.
2. **High-dimensional symbolic regression** — K=10 integer keypoints → M×14 integer actions is a hard mapping with no closed-loop OBS.
3. **Quantization** — `bins=32` smooths motion but wrong codes still decode to valid trajectories that miss the cube.
4. **Single-frame OBS** — anchors only on t=0, no visual feedback over the 400-step rollout.
5. **Model scale** — local 26B class models tend to copy a single demo's tail.

Mitigations supported by the codebase: more demos (`scripts/evaluate.py sweep` covers 5/10/20), more keypoints (K=20), fewer waypoints (M=10), stronger LLM via `--model …`.

---

## 7. Repository map (after cleanup)

```
act_kat/
├── episodes.py            HDF5 helpers
├── vision_tokens.py       DINO ViT-S/16 + FPS + patch->pixel
├── keypoint_anchors.py    paper-style anchor selection + tokenization
├── action_tokens.py       14-D quantizer + TRAJ/WP formatter
├── ollama_client.py       Ollama HTTP + artifact logging
├── replay.py              sim replay + linear/hold upsampling
└── icl.py                 prompt + generate + parse + replay

scripts/
├── download_dino_weights.py  download DINO ViT-S/16 (Meta CDN, ~83MB)
├── make_subsets.py           build demos_5/10/20 + test_5 from base
├── show_tokens.py            print OBS+ACT tokens, optional KP overlay PNG
├── replay_response.py        decode one LLM response + replay
├── gt_replay.py              decode-free upper bound on a folder
└── evaluate.py               Step 5 entry: `run` / `sweep` / `aggregate`

reports/
└── P5_KAT_ICL_Report.md      this file

P5_KAT_ICL_Complete.ipynb     single notebook covering Steps 1–5
```

---

## 8. References

1. Di Palo & Johns. *Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics.* RSS 2024.
2. Zhao et al. *ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.*
3. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers (DINO).*
