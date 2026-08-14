# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

UAV-Flow Colosseo (NeurIPS 2025): a benchmark for instruction-conditioned UAV control. A VLA policy is finetuned on UAV trajectories, served over HTTP, and evaluated by flying it in an UnrealZoo/UnrealCV simulator and scoring the resulting trajectory with nDTW.

Three subprojects, two conda environments:

| Directory | Env | Role |
|---|---|---|
| `OpenVLA-UAV/` | `openvla` (py3.10) | Fork of upstream OpenVLA. Finetuning + Flask inference server. |
| `UAV-Flow-Eval/` | `unrealcv` (py3.11) | Fork of UnrealZoo Gym. Drives the simulator against the server, computes nDTW. |
| `dataset_tools/` | either | One-off script converting HF parquet datasets to the on-disk folder format. |

The two environments are mutually incompatible (`gym==0.10.9` vs. the OpenVLA stack) and are meant to run as **two separate processes**, talking over HTTP.

## Commands

```bash
# Lint (OpenVLA-UAV only; black + ruff, line-length 121)
cd OpenVLA-UAV && make check        # check without modifying
cd OpenVLA-UAV && make autoformat   # fix in place

# Data prep: edit train_dir/parquet_dir at the bottom of the file first
cd dataset_tools && python prepare_data.py

# Finetune (edit all placeholder paths in the .sh first; 8-GPU torchrun)
cd OpenVLA-UAV && bash vla-scripts/finetune_uav.sh

# Serve (edit cfg["model_path"] in main() first; defaults to port 5007)
cd OpenVLA-UAV && python vla-scripts/openvla_act.py

# Evaluate, in a second terminal + second env
cd UAV-Flow-Eval && python batch_run_act_all.py [--server_port 5007]
cd UAV-Flow-Eval && python metric.py    # writes ./metric.txt, does not print to stdout
```

There is no test suite and no linting configured for `UAV-Flow-Eval`.

## Architecture

### The pipeline

`prepare_data.py` → per-trajectory folders (`<id>/000000.jpg…`, `log.json`) → `SimpleVLADataset` → `finetune_uav.py` (LoRA) → checkpoint → `openvla_act.py` (Flask `/predict`) ← HTTP ← `batch_run_act_all.py` (UnrealCV) → `results/<env_id>/openvla/*.json` → `metric.py` (nDTW vs. `test_jsons/`).

### Three cross-process contracts

These are the things that silently produce garbage rather than errors when broken. All three span files that don't import each other.

**1. The prompt string.** `uav_dataset.py:235` builds the training prompt and `openvla_act.py:77` rebuilds it at inference. Both must produce byte-identical text, including the `Current State: {x,y,z,yaw}` proprio formatting (`round(x, 1)`, comma-joined, no spaces). Change one, change the other.

**2. The `unnorm_key`.** Training writes action statistics into the checkpoint's `norm_stats` under the literal key `"sim"` (`update_model_norm_stats` in `finetune_uav.py`, called both before training and at every save). The server reads them back with `cfg["unnorm_key"] = "sim"`. The key is hardcoded on both sides regardless of whether the data is real or simulated.

**3. The action frame.** The model emits a 4-DoF action `[dx, dy, dz, dyaw]` **in the current pose's local frame**, yaw in radians, positions in Unreal centimeters. It is un-transformed in two stages:
- `openvla_act.py` rotates by the current yaw and adds the current position, returning a pose in the *first-frame* frame (also returns the untouched prediction as `action_ori`).
- `batch_run_act_all.py:319` (`transform_to_global`) rotates by `initial_yaw` and adds `initial_pos` to reach world coordinates, and subtracts 180° from yaw before `set_rotation` — the drone asset's forward axis is flipped relative to the convention used everywhere else.

`_transform_to_local_frame` in `uav_dataset.py` is the inverse of stage one and defines the convention.

### Coordinate and unit conventions

**See [DATA.md](./DATA.md) for the full, empirically verified data contract** — field layouts, frames, action order, and several load-bearing data quirks. The essentials:

Units differ between the two halves of the project: the **sim** data, `test_jsons/`, and the simulator are in **Unreal centimeters** (`metric.py` divides by 100 to score in meters), but the **real** UAV-Flow logs are in **meters**. Since `unnorm_key` is hardcoded to `"sim"` on both sides, mixing them fails silently rather than erroring — a real-data-trained policy moves ~100× too little in sim.

Proprio yaw is **degrees**, action yaw is **radians** — `uav_dataset.py:142` converts only the trajectory used to derive actions, deliberately leaving proprio in degrees. The frame is x-forward, y-right, z-up, yaw positive clockwise (Unreal convention).

`preprocessed_logs` rows are 6D `[x,y,z,roll,yaw,pitch]`, but **real `raw_logs` rows are 7D** — a trailing Unix timestamp the code never reads. Both work because every consumer slices `[0,1,2,4]` to get `[x,y,z,yaw]`. Note the axis order: yaw is index **4**, not 3.

Proprio and actions come from different sources in different frames: proprio from `preprocessed_logs` (first-frame frame), actions derived from `raw_logs` (current-frame local deltas).

### Training specifics

`SimpleVLADataset` (`prismatic/vla/datasets/uav_dataset.py`) is the UAV-specific replacement for upstream's RLDS pipeline — `finetune_uav.py` is a fork of `finetune.py` with RLDS swapped out. It's an `IterableDataset`, but it eagerly loads every episode's metadata and computes action statistics in `__init__`, so startup cost and memory scale with dataset size. Actions are normalized to [-1,1] against the 1st/99th percentile. First and last frames of each episode are duplicated 5× (`last_frame_repeat_count`) to bias the policy toward starting and stopping cleanly; the last action of every episode is forced to zero.

Everything else under `prismatic/` and `vla-scripts/` is upstream OpenVLA, largely unused by this project. Prefer touching the `_uav` variants.

### Evaluation specifics

The simulator needs the packaged UnrealZoo binary; `gym_unrealcv/envs/setting/Track/DowntownWest.json` must have `env_bin_win` pointed at it. The upstream README notes this was tested on Windows, and `metric.py` still uses Windows-style relative paths (`r'.\results\...'`).

`control_loop` terminates on any of: `max_steps` (default 100), 10 consecutive near-zero actions (`ACTION_SMALL_STEPS`), or a `done: true` from the server. The stall check is what normally ends a task, since the policy learns to emit zeros when finished.

`batch_run_act_all.py` skips any task whose `_2d.png`/`_3d.png` already exist, so **delete the output directory to force a re-run**. It also POSTs to `/reset` before each task; `openvla_act.py` doesn't implement that route, and the resulting failure is logged and ignored.

`metric.py` scores per instruction class (from `classified_instr.json`: Turn, Move, Shift, Rotate, Surround, Ascend/Descend, Approach, Retreat, Pass, Land) with class-dependent handling: rotation-only classes (Turn, Rotate) zero out position so only orientation is scored, and Turn/Move sample every 2 frames instead of 5. Ground truth comes from `reference_path_preprocessed` in `test_jsons/*.json`, capped at 20 points.

## Conventions

Paths to models, datasets, and simulator binaries are placeholders committed into source (`finetune_uav.sh`, `prepare_data.py`, `openvla_act.py`'s `cfg`) rather than being read from config or argv. Filling them in dirties the working tree; don't commit local paths back.
