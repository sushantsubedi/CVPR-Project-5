# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

### KAT-ICL milestone (Transfer Cube only)
This repo now also contains a **KAT-style in-context imitation learning (ICL)** milestone pipeline for **ACT Transfer Cube**.

Key idea: serialize **observation tokens** (top-view keypoints) and **action tokens** (quantized bimanual joint+gripper waypoints) as text, prompt a local LLM via **Ollama**, then **decode + replay** predicted actions in simulation.

#### Files added for KAT-ICL
Package: `act_kat/`
- `episode_schema.py`: HDF5 schema inspector + JSONL reports
- `episodes.py`: episode id listing and HDF5 frame/env helpers
- `vision_tokens.py`: top-view keypoint tokens (ViT-DINO via `timm`, optional depth)
- `action_tokens.py`: 14D action quantization + `ACT_START … ACT_END` formatting
- `icl.py`: shared ICL prompt/generate/parse/decode/replay logic (JSON-constrained Ollama output)
- `prompting.py`: few-shot prompt assembly
- `ollama_client.py`: Ollama `/api/generate` client + artifact logging
- `replay.py`: sim replay + piecewise-constant waypoint upsampling

Scripts (used by `kat_icl_transfer_cube_report.ipynb`):
- `schema_report.py`, `make_ds10_subsets.py`, `tokenize_obs.py`, `tokenize_action.py`
- `visualize_keypoints.py`, `run_icl.py`, `replay_from_icl_json.py`, `replay_manual_test.py`, `eval_icl_json_folders.py`

Notebook: `kat_icl_transfer_cube_report.ipynb`

Artifacts: `artifacts/prompts/*.txt`, `artifacts/responses/*.json`, `replays/`, `results_icl_json*.csv`


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

#### KAT-ICL dependencies (if using `.venv`)
The KAT-ICL milestone code assumes the repo-local venv `./.venv` has the ACT sim deps plus:
- `requests` (Ollama HTTP calls)
- `timm` (ViT-DINO backbone for keypoint tokens)

If your `record_sim_episodes.py` runs already, you likely only need:

    source .venv/bin/activate
    python3 -m pip install requests timm

Download DINO ViT weights once (local file, no Hugging Face at inference):

    python3 scripts/download_dino_weights.py

Weights are saved to `assets/weights/vit_small_patch16_224.dino.pth` (override with `ACT_KAT_DINO_CHECKPOINT`).

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

### KAT-ICL milestone quickstart (Transfer Cube)
Run the notebook:
- Open `kat_icl_transfer_cube_report.ipynb` and execute top-to-bottom.

Or run key steps as scripts:

1) Collect + visualize demos (downsample 10):

    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/transfer_cube/ds10/demos_2_env0 --num_episodes 2 --downsample_rate 10
    python3 visualize_episodes.py --dataset_dir data/transfer_cube/ds10/demos_2_env0 --episode_idx 0

2) Monitor episode file structure (schema report):

    python3 scripts/schema_report.py --dataset_dir data/transfer_cube/ds10/demos_2_env0 --out artifacts/schema_report_ds10_env0.jsonl

3) Tokenize obs + actions:

    python3 scripts/tokenize_obs.py --episode_hdf5 data/transfer_cube/ds10/demos_2_env0/episode_0.hdf5 --k 10 --device cpu
    python3 scripts/tokenize_action.py --episode_hdf5 data/transfer_cube/ds10/demos_2_env0/episode_0.hdf5 --M 20

4) Call Ollama (JSON actions; saves `artifacts/prompts/<run_id>.txt` and `artifacts/responses/<run_id>.json`):

    python3 scripts/run_icl.py --dataset_dir data/transfer_cube/ds10/demos_2_env0 --demo_ids 0 --query_id 1 --K 10 --M 20 --model gemma4:26b --run_id my_run

5) Decode + replay from ICL JSON:

    python3 scripts/replay_from_icl_json.py --icl_json artifacts/responses/my_run.json --quantizer_json artifacts/quantizers/<quantizer>.json --bins 64 --episode_len 400 --box_pose_episode_hdf5 data/transfer_cube/subsets/test_5/episode_21.hdf5

6) Manual LLM response on one test episode (paste JSON or use a file; video under `artifacts/replays/`):

    python3 scripts/replay_manual_test.py --tests_dir data/transfer_cube/ds10/subsets_depth/test_5 --test_id 20 --demos_dir data/transfer_cube/ds10/subsets_depth/demos_5 --llm_json path/to/response.json

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

