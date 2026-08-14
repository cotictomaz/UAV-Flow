# Environment Setup — Reproducing UAV-Flow Colosseo

Working log for reproducing the paper's numbers with the released `OpenVLA-UAV` checkpoint.
Kept in sync as we go; `[ ]` = not done yet, `[x]` = done and verified.

## Topology

Two machines, split along the process boundary the code already has:

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│ Linux box (A100-40GB)       │         │ Windows box (discrete GPU)           │
│                             │  HTTP   │                                      │
│ conda env: openvla (3.10)   │◄────────┤ conda env: unrealcv (3.11)           │
│ vla-scripts/openvla_act.py  │ /predict│ batch_run_act_all.py  ──► Collection │
│ Flask, port 5007            │         │ metric.py             ◄── UnrealCV   │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

**Why this is safe** (verified in code, not assumed):

- `openvla_act.py:123` binds `host='0.0.0.0'` — already accepts remote connections.
- `/predict` (`openvla_act.py:60-115`) is stateless: image + proprio + instruction in, action out.
  Nothing is retained between calls, so the client being remote changes nothing.
- Payload is a base64 PNG resized to 224×224 plus a 4-float state (`batch_run_act_all.py:52-62`)
  — ~100 KB per step, sequential, 30 s client timeout. Fine over LAN or VPN.
- Everything simulator-side (UnrealCV, capture, `transform_to_global`, plots, `metric.py`)
  runs in the eval process and stays on Windows.

**The one blocker:** `batch_run_act_all.py:477` hardcodes the host —

```python
server_url = f"http://127.0.0.1:{args.server_port}/predict"
```

Only the port is exposed as a flag. Solved with an SSH tunnel (§4), no code change.

---

## 1. Windows: prerequisites — DONE

Verified on this machine:

| Check | Result |
|---|---|
| GPU / driver | NVIDIA RTX A4000 (8 GB), driver 580.92, CUDA 13.0 |
| Session type | `Console` (physical) — no RDP black-frame risk |
| Free disk | C: 301 GB, **D: 276 GB (DriveType 3, local)**, P:/T: (network — do not use) |
| `LongPathsEnabled` | already `1`, no change needed |
| Git | 2.55.0.windows.3 |
| Conda | 26.5.3 (installed during this step) |
| Admin rights | **none** — see below |

**Target drive is D:** — local fixed disk. P: and T: are large but network-backed; a UE
binary launched from a share loads too slowly for UnrealCV's startup timeout.

**No admin rights on this machine.** It turned out not to matter:

- `LongPathsEnabled` was already set (the only step that would have needed elevation).
- Miniconda installs per-user with `/InstallationType=JustMe`.
- UnrealCV binds loopback, and Windows does not filter loopback traffic, so no firewall
  approval is expected (see §3.5).

**Miniconda, no-admin install.** Download, then let `cmd.exe` expand the path — PowerShell
drops the backslash in `/D=`, producing `C:\Users\<user>Miniconda3` and a "not writable"
error, because that resolves to a new folder directly under `C:\Users`:

```powershell
curl.exe -L -o "$env:TEMP\Miniconda3.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
cmd /c start /wait "" "%TEMP%\Miniconda3.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
& "$env:USERPROFILE\Miniconda3\shell\condabin\conda-hook.ps1"
conda init powershell   # then reopen PowerShell
```

`/D=` must be last and unquoted — an installer quirk.

## 2. Windows: repo + conda env

The eval half of the repo has to exist on Windows too.

```powershell
cd D:\
git clone https://github.com/cotictomaz/UAV-Flow.git
cd D:\UAV-Flow\UAV-Flow-Eval

conda create -n unrealcv python=3.11 -y
conda activate unrealcv
pip install -e .
```

`pip install -e .` pulls `gym==0.10.9`, `unrealcv>=1.1.5`, `opencv-python`, `matplotlib`,
`simple_pid`, `pynput`, `docker`, `modelscope` (from `pyproject.toml`).

- [ ] Verify the install resolved:

```powershell
python -c "import gym, gym_unrealcv, unrealcv; print(gym.__version__, unrealcv.__version__)"
```

> `gym==0.10.9` is from 2018 and is being installed on Python 3.11. If it fails to import
> (typically `collections.Iterable` style breakage), see Troubleshooting.

**Editable install matters:** settings are read from the repo working tree, not a copy —
`misc.get_settingpath` resolves `os.path.dirname(gym_unrealcv.__file__) + envs/setting/<file>`
(`gym_unrealcv/envs/utils/misc.py:14-17`). So editing the JSON in the repo is the right move.

## 3. Windows: simulator binary

### 3.1 Extract

Extract `Collection_WinNoEditor_0424_25.zip` to a **short** root, e.g. `D:\UE\`.

- [ ] Confirm the executable path — the zip may or may not already contain the top folder,
      so check for accidental double-nesting:

```powershell
Get-ChildItem -Path D:\UE -Recurse -Filter Collection.exe | Select-Object FullName
```

Expected shape:

```
D:\UE\Collection_WinNoEditor_0424_25\Collection\Binaries\Win64\Collection.exe
```

- [ ] Check whether `unrealcv.ini` sits next to the exe in `...\Binaries\Win64\`. The launcher
      is expected to manage the UnrealCV port through it. **(unverified — confirm on Windows)**

### 3.2 Point the config at it

Edit `UAV-Flow-Eval/gym_unrealcv/envs/setting/Track/DowntownWest.json`, field `env_bin_win`.
It currently holds the authors' path (`D:\unrealzoo-gym-new\UnrealEnv\...`).

```json
"env_bin_win": "D:\\UE\\Collection_WinNoEditor_0424_25\\Collection\\Binaries\\Win64\\Collection.exe",
```

**Backslashes must be doubled** — it's JSON. Leave `env_bin` (Linux) and `env_map` alone;
`base_env.py:109-117` selects `env_bin_win` on `sys.platform == 'win32'`, and `env_map`
(`DowntownWest`) is what `set_map` switches to after launch.

- [ ] Validate the JSON parses and the path exists:

```powershell
python -c "import json,os; p=json.load(open(r'gym_unrealcv\envs\setting\Track\DowntownWest.json'))['env_bin_win']; print(p, os.path.exists(p))"
```

### 3.3 How launching actually works

**Do not start `Collection.exe` yourself before the eval script.** The env launches it:
`base_env.py:121` builds `RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)` and
`base_env.py:631-633` calls `ue_binary.start(...)`, which returns the `(ip, port)` that
`Character_API` then connects to. A manually-started instance won't be the one it talks to.

- [ ] Worth confirming how `RunUnreal` resolves the path (absolute vs. relative to the
      UnrealEnv dir) on your machine — the shipped config uses an absolute path, so absolute
      is the safe choice:

```powershell
python -c "import inspect, unrealcv.launcher as l; print(inspect.getsource(l.RunUnreal))"
python -c "import unrealcv.util as u; print(u.get_path2UnrealEnv())"
```

### 3.4 Textures are *not* needed

`load_env.py -e Textures` is unnecessary here. `get_textures()` is only reached when
`'track_train' in env_name` (`augmentation.py:15-21`), and our env is `DowntownWest`.
The env id ends in `v0` → `reset_type == 0`, so `environment_augmentation` never fires either.

### 3.5 Smoke test — simulator alone, no server

This mirrors `batch_run_act_all.py:479-490` exactly, so it isolates simulator problems
from server/network problems. Run from `UAV-Flow-Eval\`:

```python
# smoke_sim.py
import gym, gym_unrealcv
from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

env = gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0')
env = time_dilation.TimeDilationWrapper(env, 10)
env.unwrapped.agents_category = ['drone']
env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
env.seed(0)
env.reset()
env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
print(env.unwrapped.unrealcv.get_camera_config())
env.close()
```

- [ ] A game window opens, loads DowntownWest, prints a camera config, exits cleanly.
- [ ] Firewall: no prompt expected. UnrealCV binds loopback and Windows does not filter
      loopback traffic. If a dialog does appear and can't be approved (no admin on this
      machine), check what IP `ue_binary.start()` actually returned before assuming it's
      the firewall.

The env id decomposes as `UnrealTrack-<map>-<action><obs>-v<reset_type>` and registers
`gym_unrealcv.envs:Track` with `env_file=Track/DowntownWest.json` (`gym_unrealcv/__init__.py:138-166`)
— which is why §3.2 edits that specific file.

## 4. Networking: Linux server ← Windows client

From **Windows**, with the Flask server already running on Linux:

```powershell
ssh -N -L 5007:localhost:5007 <user>@<linux-host>
```

`batch_run_act_all.py` then reaches it unmodified at `127.0.0.1:5007`. This also avoids
exposing the Flask dev server (no auth, no TLS, single-threaded) on the network.

- [ ] Tunnel up, and `curl http://127.0.0.1:5007/predict` from Windows returns something
      other than a connection error.

Alternative if the tunnel is inconvenient: add a `--server_host` arg alongside
`--server_port` and edit line 477. Two lines, but it dirties the working tree.

## 5. Linux: inference server

- [ ] `conda activate openvla` (env already exists at `/opt/conda/envs/openvla`).
- [ ] Download the released checkpoint: `wangxiangyu0814/OpenVLA-UAV` on HuggingFace.
- [ ] Set `cfg["model_path"]` in `OpenVLA-UAV/vla-scripts/openvla_act.py:156`.
      Leave `unnorm_key = "sim"` — it is hardcoded on both sides regardless of data source.
- [ ] `python vla-scripts/openvla_act.py` → listens on 5007.

Loads with `flash_attention_2` + bf16 on `gpu_id: 0`. The A100-40GB is plenty for the 7B model.

## 6. Run the evaluation

```powershell
# Windows, conda env unrealcv, tunnel up, server up
cd D:\UAV-Flow\UAV-Flow-Eval
python batch_run_act_all.py
```

Defaults (`batch_run_act_all.py:428-436`): env `UnrealTrack-DowntownWest-ContinuousColor-v0`,
time dilation 10, seed 0, tasks from `./test_jsons` (**273 files**), output to
`./results/UnrealTrack-DowntownWest-ContinuousColor-v0/openvla`, port 5007, `max_steps` 100.

Then:

```powershell
python metric.py     # writes .\metric.txt — prints nothing to stdout
```

`metric.py:247-250` reads `.\results\UnrealTrack-DowntownWest-ContinuousColor-v0\openvla`
against `.\test_jsons` — matching the eval defaults exactly, so no path edits needed as long
as both run from `UAV-Flow-Eval\` on the same machine.

## 7. Behaviors that look like bugs but aren't

- **`/reset` fails every task.** `batch_run_act_all.py:408` POSTs to `/reset`; the server
  doesn't implement that route. The failure is logged and ignored.
- **Tasks get skipped on re-run.** Any task whose `_2d.png` and `_3d.png` already exist is
  skipped (`batch_run_act_all.py:513-517`). Delete the results directory to force a re-run.
- **Episodes end early.** `control_loop` stops on `max_steps`, a `done` from the server, or
  10 consecutive near-zero actions. The stall check normally ends a task — the policy is
  trained to emit zeros when finished.
- **Two extra actors spawn at startup.** A character and a car are placed at fixed positions
  before the task loop (`batch_run_act_all.py:500-508`). Intentional.
- **`metric.py` produces no console output.** It redirects stdout to `metric.txt`.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Binary never launches, or path error at `env.reset()` | `env_bin_win` wrong or backslashes not doubled. Re-run the §3.2 check. |
| UnrealCV connection refused / timeout on first reset | Binary crashed or is still loading. `launch_ue_env` passes `sleep_time=10` (`base_env.py:633`); on a slow disk raise it. Try launching the exe by hand once to confirm it renders. |
| Black or empty frames | GPU not really attached (RDP), or driver issue. Use a physical/GPU-backed session. |
| `gym==0.10.9` won't install or import on 3.11 | Try `pip install setuptools<60 wheel` first. If it still breaks, drop the eval env to Python 3.10 — nothing in `UAV-Flow-Eval` needs 3.11. |
| Server reachable locally but not from Windows | Tunnel down, or you edited the URL instead of tunneling. Server already binds `0.0.0.0`, so it's the client side. |
| Trajectories are ~100× too small | `unnorm_key` mismatch — a checkpoint trained on the **real** (meters) data being scored in Unreal centimeters. The released sim checkpoint is the right one for this benchmark. |
| `metric.py` scores far fewer than 273 tasks | It silently skips any task with no matching result file. Check the results dir is complete. |

## 9. Do not commit

Per repo convention, model/dataset/simulator paths are placeholders committed into source.
Filling in `env_bin_win` and `cfg["model_path"]` dirties the tree — **don't commit them back**.

## 10. Record for reproducibility

- [ ] Windows GPU + driver version
- [ ] `gym` / `unrealcv` / `numpy` versions in the `unrealcv` env
- [ ] Checkpoint revision (HF commit hash) used for `model_path`
- [ ] `metric.txt` from the completed run