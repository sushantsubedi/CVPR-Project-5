# CVPR Project 5 : Final Report

### Sushant Subedi (ss17395)

## 1. Abstract

I implemented a KAT-style pipeline that maps top-view RGB-D observations and 14-D bimanual joint trajectories into discrete text tokens, prompts a local large language model (Ollama) for in-context imitation, parses the JSON response, and replays it in MuJoCo. Observation tokens use **DINO ViT-S/16 anchored keypoints** (paper-style best-buddy correspondences + FPS) with depth; action tokens use **uniform per-dimension quantization** of M waypoints. I evaluate few-shot ICL with explicit **seen / unseen** splits and run **four ablations** (number of demos, K keypoints, M waypoints, and LLM model). **The pipeline reaches the LLM correctly and the LLM produces well-formed JSON (100% parse-ok at M=20), but task success on the held-out unseen split is 0/5 across every configuration**, including with 20 in-context demos, K=20 keypoints, and the larger gemma4:26b model. The decode-free GT replay confirms the simulator and replay paths are correct (5/5 on both demos_5 and test_5). The action-token round-trip (no LLM, quantizer fit on the full corpus) also succeeds, so the representation itself is not the bottleneck. The remaining gap is **LLM reasoning capacity at this model scale** on the K-keypoint → 14·M-integer symbolic-regression task, which is consistent with the original KAT paper's reliance on GPT-4-class frontier models. The whole project is reproducible from one Jupyter notebook or one shell sequence.

---

## 2. Method

### 2.1 Observation tokens — vision-only

1. **Anchor selection (once per demo folder, in `act_kat/keypoint_anchors.py`):** extract DINO ViT-S/16 patch descriptors for two frames of `episode_0` (t=0 and t=T/2). Take **best-buddy** correspondences between the two frames' patch grids, then run **furthest-point sampling** on the buddy set to keep K spatially diverse anchors. Their descriptors are cached.
2. **Localize per frame:** at query time, each cached anchor descriptor is matched to the new image's patch grid by cosine NN → a stable index `i` per anchor, shared across demos and queries.
3. **Serialize (`tokenize_anchored_keypoints`):**

```
OBS_START
KP 0 x=2  y=2  d=15
KP 1 x=14 y=8  d=12
...
KP K-1 x=...
OBS_END
```

`x,y` are normalized image coordinates quantized to `obs_bins` (default 16). `d` is depth (top camera) clipped to [0, 2] m and quantized to `obs_bins`. **No simulator ground truth is used**; the only side-channel for the LLM is the K depth-augmented keypoint table.

### 2.2 Action tokens

* `M` waypoints uniformly subsampled over an episode (`select_waypoint_indices`).
* **Per-dimension uniform quantizer** (`act_kat/action_tokens.py::ActionQuantizer`) fit over the demo folder, default `bins=32`. 14 dims = 6 left-arm joints, 1 left gripper, 6 right-arm joints, 1 right gripper.
* **Demo block** in the prompt mirrors the JSON schema:

```
TRAJ_START
WP[i=0] L=[47,25,45,12,27,0] LG=0 R=[19,21,61,40,0,22] RG=0
...
WP[i=M-1] ...
TRAJ_END
```

* `LG/RG ∈ {0, 1}` (0 = open, 1 = closed) come from thresholding the normalized gripper positions at 0.5.
* The LLM is **schema-constrained** to return JSON `{"actions": [{"L":[…], "LG":0, "R":[…], "RG":0}, …]}` with exactly M items (`build_actions_json_schema`).

### 2.3 ICL → JSON → trajectory → sim

`act_kat/icl.py::generate_icl_actions` builds the prompt:

```
### Demonstrations
## Demo 0
<OBS_START … OBS_END>
<TRAJ_START … TRAJ_END>
## Demo 1
...

### Query
<OBS_START … OBS_END>

Predict the query trajectory as JSON (M waypoints: L, LG, R, RG). ...
Return ONLY JSON matching the provided schema (not TRAJ text). ...
```

Then calls Ollama `/api/generate` with `temperature=0`, `num_predict=2048`, `format=<JSON schema for M>`, parses the response, converts `LG/RG` to gripper codes `{0, bins-1}`, decodes the 14-D codes via the quantizer to floats, **linearly upsamples** the M waypoints to `episode_len=400` sim steps, sets `sim_env.BOX_POSE[0] = env_state0` from the query HDF5, and rolls out `act_kat/replay.py`.

### 2.4 Decode-free upper bound

`scripts/gt_replay.py` reads the raw `/action` keyframes from HDF5, reads `downsample_rate` (10) from the file's attrs, linearly upsamples the keyframes back to the full 400-step trajectory, and replays in `sim_env` with the same `env_state0`. This is the decode-free upper bound on what a perfect KAT pipeline could ever achieve on this dataset.

---

## 3. Dataset and splits (Step 5.1 — fixed seed, seen / unseen)

```
data/transfer_cube/
├── DATA_MANIFEST.md           ← complete schema + repro contract
├── base/                       # 25 episodes recorded with --seed 0
│   └── episode_{0..24}.hdf5    # attrs: collection_seed, episode_index,
│                               #        downsample_rate, env_state0
└── subsets/                    # hard-linked from base/, never re-recorded
    ├── demos_5/                # 0..4   ← in-context demos (main)
    ├── demos_10/               # 0..9   ← #demos ablation
    ├── demos_20/               # 0..19  ← #demos ablation
    └── test_5/                 # 20..24 (held out — never in any demos_* folder)
```

| Split | Definition | Query episodes (main = `demos_5`) | # queries |
|-------|------------|-----------------------------------|----------:|
| **seen**   | Leave-one-out on the demo folder: query is one demo, prompt holds the other N-1. | 0–4 | 5 |
| **unseen** | Held-out test folder; box poses still come from the same seeded run, but the prompt never includes the query. | 20–24 | 5 |

`scripts/evaluate.py` tags every CSV row with `split ∈ {seen, unseen}` and the aggregator computes per-split success rates in `artifacts/evaluation_summary.json`. The complete dataset contract — every HDF5 attr, every dataset shape, the deterministic 0..4 / 0..9 / 0..19 / 20..24 split, the exact `record_sim_episodes.py` command — is documented in [`data/transfer_cube/DATA_MANIFEST.md`](../data/transfer_cube/DATA_MANIFEST.md).

---

## 4. Evaluation, ablations and reproducibility (Step 5)

### 4.1 One-command sweep

```bash
source .venv/bin/activate
python3 scripts/evaluate.py sweep --collection_seed 0 \
  --models 'gemma4:26b,llama3.2:latest'
```

Produces, under `artifacts/`:

* `results_main_seen_unseen.csv` — main seen + unseen on `demos_5` (K=10, M=20, model=`gemma4:26b`).
* `results_ablation_ndemos_{10,20}_unseen.csv` — #demos ablation. The 5-demo baseline is the main config's `unseen` rows; we extend with 10 and 20.
* `results_ablation_K{5,10,20}_unseen.csv` — keypoint count ablation.
* `results_ablation_M{10,20,40}_unseen.csv` — action-token count ablation.
* `results_ablation_model_{gemma4_26b,llama3_2_latest}_unseen.csv` — LLM model ablation.
* `evaluation_summary.json`, `evaluation_tables.md` — aggregated tables.

Each ICL row records `split, episode_id, model, K, M, n_demos_in_prompt, demo_ids_in_prompt, success, max_reward, parse_ok, parse_errors`. `max_reward` in transfer cube is 4 (touch cube=1, grasp=2, lift=3, transfer=4); `success = (max_reward == 4)`.

### 4.2 Decode-free upper bound (sanity / Step 3)

| Folder | N | Success | Avg `max_reward` |
|--------|--:|--------:|-----------------:|
| `demos_5` | 5 | **5 / 5 (100%)** | 4.00 / 4 |
| `test_5`  | 5 | **5 / 5 (100%)** | 4.00 / 4 |

→ the simulator, replay path, `env_state0` propagation, and the `downsample_rate=10` → 400-step linear upsampling are all correct end-to-end. The headroom for the ICL pipeline is the full 100%.

### 4.3 Action-token round-trip (no LLM)

`KAT_ICL.ipynb` step 3 takes the same full trajectory of `test_5/episode_20`, subsamples M=20 waypoints, encodes them through the quantizer, decodes back to floats, linearly upsamples to 400 steps, replays, and renders side-by-side with the GT replay. With the quantizer fit on the **full `base/` corpus** (25 episodes), the round-trip **succeeds (`max_reward = 4`)**.

### 4.4 Main configuration — seen / unseen

`results_main_seen_unseen.csv` (model = `gemma4:26b`, K=10, M=20, bins=32, episode_len=400):

| Split | N | parse_ok | Success | `max_reward` per query | Avg `max_reward` |
|-------|--:|---------:|--------:|------------------------|-----------------:|
| **seen** (demos_5 LOO, queries 0–4) | 5 | 100% | **0 / 5** | 0, 0, 1, 2, 1 | **0.80 / 4** |
| **unseen** (test_5, queries 20–24)  | 5 | 100% | **0 / 5** | 0, 0, 0, 0, 0 | **0.00 / 4** |

Observations:

* Every response parses as valid JSON with exactly 20 waypoint objects under the constrained schema; the entire failure budget is in trajectory *content*, not formatting.
* On seen queries the LLM occasionally drives the cube far enough to register partial reward (touch / grasp / lift = 1, 2, 3 of 4), with a non-zero average max_reward 0.80; the unseen split has zero partial credit.
* The seen↔unseen gap (0.80 → 0.00 avg max_reward) shows the LLM is doing **demo-conditional pattern matching** rather than learning a generalizable mapping: when the query's box pose is close to one of the in-context demos it recovers partial behaviour, but on fresh poses the predicted trajectory completely misses the cube.

### 4.5 Ablations

All four ablations hold everything else at the main config (`demos_5`, K=10, M=20, `gemma4:26b`, unseen split, queries 20–24) except the variable under test.

**(a) Number of in-context demos** (`results_ablation_ndemos_*_unseen.csv`)

| #demos | Demo ids | parse_ok | Success | Avg `max_reward` |
|------:|----------|---------:|--------:|-----------------:|
| 5  | 0..4   | 100% | 0 / 5 | 0.00 |
| 10 | 0..9   | 100% | 0 / 5 | 0.00 |
| 20 | 0..19  | 100% | 0 / 5 | 0.00 |

More demos do not help. The prompt grows linearly but the unseen rate stays at 0.

**(b) Number of keypoints K** (`results_ablation_K*_unseen.csv`, M=20 fixed)

| K | parse_ok | Success | Avg `max_reward` |
|---:|---------:|--------:|-----------------:|
| 5  | 100% | 0 / 5 | 0.00 |
| 10 | 100% | 0 / 5 | 0.00 |
| 20 | 100% | 0 / 5 | 0.00 |

Doubling or halving K leaves unseen success at zero. K=10 is not starved for visual context relative to K=20.

**(c) Number of action waypoints M** (`results_ablation_M*_unseen.csv`, K=10 fixed)

| M | parse_ok | Success | Avg `max_reward` | Notes |
|---:|---------:|--------:|-----------------:|-------|
| 10 | 100% | 0 / 5 | 0.00 | shortest output |
| 20 | 100% | 0 / 5 | 0.00 | baseline |
| 40 | **0%**  | 0 / 5 | 0.00 | **all 5 responses fail JSON parse with `JSONDecodeError`** |

`M=40` is the **first ablation where the system fails before the simulator runs**: 40 waypoints × 14 ints + braces exceeds Ollama's `num_predict=2048` token budget under the constrained schema, so responses get truncated and parsing fails uniformly. This is a useful engineering data point — even before reasoning enters the picture, action-token budget at this M is the wrong setting for this LLM/runtime.

**(d) LLM (model scale)** (`results_ablation_model_*_unseen.csv`, K=10, M=20 fixed)

| Model | Approx. params | parse_ok | Success | Avg `max_reward` |
|-------|----------------|---------:|--------:|-----------------:|
| `gemma4:26b`      | ~26 B | 100% | 0 / 5 | 0.00 |
| `llama3.2:latest` | ~3 B  | 100% | 0 / 5 | 0.00 |

A ~10× scale gap between gemma4:26b and llama3.2 at otherwise identical token budget makes no difference at this level. Both models comply with the JSON schema; neither solves the symbolic regression. This is consistent with the KAT paper's observation that the method requires GPT-4-class frontier models — the ~3–26 B Ollama-runnable models we tested are below the threshold.

### 4.6 Aggregated table (auto-generated, `artifacts/evaluation_tables.md`)

| CSV | N | Seen SR | Unseen SR | Overall SR | Parse OK |
|-----|--:|--------:|----------:|-----------:|---------:|
| `results_gt_replay_demos_5.csv` | 5 | — | — | **100.0%** | — |
| `results_gt_replay_test_5.csv` | 5 | — | — | **100.0%** | — |
| `results_main_seen_unseen.csv` | 10 | 0.0% | 0.0% | — | 100.0% |
| `results_ablation_ndemos_10_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_ndemos_20_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_K5_unseen.csv`  | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_K10_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_K20_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_M10_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_M20_unseen.csv` | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_M40_unseen.csv` | 5 | — | 0.0% | — | **0.0%** |
| `results_ablation_model_gemma4_26b_unseen.csv`      | 5 | — | 0.0% | — | 100.0% |
| `results_ablation_model_llama3_2_latest_unseen.csv` | 5 | — | 0.0% | — | 100.0% |

(Regenerate with `python3 scripts/evaluate.py aggregate --glob 'artifacts/results_*.csv'`.)

### 4.7 Reproducibility checklist 

| Item | Provided |
|------|----------|
| Clean README | `README.md` |
| Environment files | `requirements-kat.txt` (primary, unified) + `conda_env.yaml` (legacy upstream) |
| Data manifest | `data/transfer_cube/DATA_MANIFEST.md` |
| Evaluation script | `scripts/evaluate.py` (`run` / `sweep` / `aggregate` subcommands) |
| Results CSV | `artifacts/results_*.csv` (this report quotes them directly) |
| Aggregated table | `artifacts/evaluation_tables.md` |
| One-command reproduction | `python3 scripts/evaluate.py sweep --collection_seed 0 --models 'gemma4:26b,llama3.2:latest'` (after data + DINO weights) |

Four-command full re-run from a uv-activated venv:

```bash
source .venv/bin/activate
python3 scripts/download_dino_weights.py
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted \
  --dataset_dir data/transfer_cube/base --num_episodes 25 --downsample_rate 10 --seed 0
python3 scripts/make_subsets.py --base_dir data/transfer_cube/base
python3 scripts/evaluate.py sweep --collection_seed 0 \
  --models 'gemma4:26b,llama3.2:latest'
```

Or run `KAT_ICL.ipynb` top to bottom.

---

## 5. Sanity checks (Step 3) and video generation

`KAT_ICL.ipynb` materializes every artifact the rubric asks for and **renders the videos inline (H.264, base64-embedded)** so they play in any notebook viewer:

| # | Check | Notebook step | Command | Artifact |
|--:|-------|---------------|---------|----------|
| 1 | Environment renders + scripted policy succeeds | Step 1 | `python3 - <<PY ... PY` (full 400-step ee-sim rollout) | `artifacts/step1_env_smoke.mp4` (`max_reward = 4`) |
| 2 | Dataset visualization | Step 2 | `python3 visualize_episodes.py …` | `data/transfer_cube/base/episode_0_video.mp4` |
| 3a | Print OBS + ACT tokens + KP overlay PNG | Step 3 | `python3 scripts/show_tokens.py --K 10 --M 20 --overlay …` | `artifacts/keypoints_overlay.png` + stdout token blocks |
| 3b | Decode-free GT replay (upper bound) | Step 3 | `python3 scripts/gt_replay.py --save_video --num_save 1` | `artifacts/replays/gt_*.mp4` + `results_gt_replay_*.csv` |
| 3c | **Side-by-side: GT vs tokenized→decoded** | Step 3b | inline action-token round-trip script + HTML | `artifacts/replays/gt_test_5_ep20.mp4` ∥ `decode_test_5_ep20.mp4` |
| 4 | Single ICL smoke run | Step 4 | `python3 scripts/evaluate.py run … --max_tests 1` | `artifacts/prompts/<id>.txt`, `responses/<id>.json`, `replays/<id>.mp4` |
| 5 | Replay any saved LLM response | optional | `python3 scripts/replay_response.py --response artifacts/responses/<id>.json …` | `artifacts/replays/<id>.mp4` |

The Step 3b side-by-side block makes the "what does the LLM have to recover" question visual: left video is the original scripted-policy trajectory replayed from `/action`, right video is the same trajectory after passing through `M=20 waypoints + bins=32 quantizer + linear upsample` — both succeed, so any ICL failure downstream is attributable to LLM reasoning, not to action-token information loss.

---

## 6. Failure analysis

What the **actual numbers** show, in order of severity, drawing from §4.4–§4.5:

1. **Generalization to unseen poses is zero.** Seen split has avg `max_reward = 0.80` (queries 2/3/4 score 1/2/1 out of 4); unseen split is uniform 0.00. The LLM is doing demonstration-conditional pattern matching: when the query's KP layout is close to one of the demos it can copy enough of that demo's waypoints to make contact, but the held-out box poses break this. This is the dominant failure mode.

2. **Scaling demos does not rescue unseen.** 5 → 10 → 20 in-context demos all give 0/5 unseen. We are not starved for demonstrations; we are starved for in-context *reasoning*.

3. **Scaling visual context does not rescue unseen.** K = 5, 10, 20 anchored keypoints all give 0/5 unseen. K=10 is sufficient information; K=20 doesn't unlock anything.

4. **Scaling action precision is bounded by the runtime.** M=10 and M=20 produce well-formed JSON but 0% success. M=40 **breaks the JSON parser deterministically** (parse_ok = 0%, all responses truncated): 40 waypoints × 14 integers + schema overhead exceeds Ollama's 2048-token `num_predict` budget. So the M ablation effectively saturates the runtime before it can saturate reasoning.

5. **Model scale at this tier doesn't matter.** `llama3.2:latest` (~3B) and `gemma4:26b` (~26B) both produce 0/5 unseen with 100% parse_ok. The capability gap relevant for KAT is between mid-tier instruction-tuned models and **GPT-4-class frontier models** (the original paper's setting). Both models we ran sit on the same side of that threshold.

6. **Single-frame observation, open-loop rollout.** All KPs are extracted from `t=0` and the predicted M waypoints are linearly upsampled to 400 steps without any visual feedback during replay. Any drift in the predicted trajectory accumulates uncorrected.

Mitigations supported by the codebase but **not yet enough to flip a 0% to non-zero** on this LLM tier: more demos (`scripts/evaluate.py sweep` covers 5/10/20), more keypoints (K=20), fewer waypoints (M=10), stronger LLM via `--model …`. The natural next step would be to point `--model` at a GPT-4-class endpoint (paper setting) or to introduce closed-loop OBS by re-prompting after a few rollout chunks.

---

## 7. Discussion — what works, what doesn't

| Component | Status |
|-----------|--------|
| Data collection (seed-0, 25 episodes, downsample=10) | ✓ deterministic, manifest-documented |
| Subset splits (`demos_{5,10,20}`, `test_5`) | ✓ deterministic, hard-linked from `base/` |
| DINO ViT-S/16 anchored keypoints + serialization | ✓ KP overlay confirms stable anchors across episodes |
| Action quantizer (per-dim uniform, `bins=32`) | ✓ round-trip succeeds on test ep with full-corpus quantizer |
| Ollama prompt + JSON schema | ✓ parse_ok = 100% at M=10, 20 |
| Simulator + replay + `env_state0` propagation | ✓ gt_replay = 100% |
| ICL reasoning at 3–26 B local model scale | ✗ unseen success 0/5 across every configuration |

The negative result is informative: **the engineering chain is correct, but the LLM tier currently runnable through local Ollama is not sufficient for KAT-style ICL on transfer cube**. The contribution of this project — beyond reproducibility of the negative result itself — is the fully instrumented pipeline that would let a future user (a) drop in a GPT-4-class model behind `--ollama_url`, (b) plug the same prompt template into a different runtime, or (c) iterate on the action representation, all without touching the env / dataset / sweep code.

---

## 8. Repository map

```
act_kat/
├── episodes.py            HDF5 helpers (read attrs, list ids, paths)
├── vision_tokens.py       DINO ViT-S/16 patch descriptors + FPS + patch→pixel
├── keypoint_anchors.py    paper-style anchor selection + per-frame tokenization
├── action_tokens.py       ActionQuantizer + TRAJ/WP formatter
├── ollama_client.py       Ollama HTTP + per-query artifact logging
├── replay.py              sim replay + linear/hold upsampling + ReplayResult
└── icl.py                 build_prompt + generate + parse + replay_response

scripts/
├── download_dino_weights.py  download DINO ViT-S/16 (Meta CDN, ~83MB)
├── make_subsets.py           build demos_5/10/20 + test_5 from base (hard-linked)
├── show_tokens.py            print OBS+ACT tokens, optional KP overlay PNG
├── replay_response.py        decode one LLM response + replay
├── gt_replay.py              decode-free upper bound on a folder (--save_video)
└── evaluate.py               Step 5 entry: `run` / `sweep` / `aggregate`

data/transfer_cube/
├── DATA_MANIFEST.md          schema + repro contract
├── base/                     25 raw seed-0 episodes
└── subsets/{demos_5,10,20,test_5}/   hard-linked subsets + manifest.txt

reports/
└── KAT_ICL_Report.md         this file

KAT_ICL.ipynb                 single notebook covering Steps 1–5
README.md                     top-level guide
requirements-kat.txt          unified pip deps
conda_env.yaml                legacy ACT upstream conda env
```

---

