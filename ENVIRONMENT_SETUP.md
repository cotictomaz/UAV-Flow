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

**The one blocker:** `batch_run_act_all.py:477` hardcoded the host —

```python
server_url = f"http://127.0.0.1:{args.server_port}/predict"
```

Only the port was exposed as a flag. **Update:** the Linux box turned out to be a Kubeflow
notebook pod (cluster `IVI-AWZ`), not an SSH-reachable host — no `sshd`, reached instead via
a kubeconfig + the VS Code Kubernetes extension. Plan changed from an SSH tunnel to a
`kubectl`-style port-forward (§4). Also added a `--server_host` flag (default `127.0.0.1`,
so default behavior is unchanged) as a documented escape hatch in case port-forwarding is
ever inconvenient — not currently needed since the port-forward maps straight to
`127.0.0.1` anyway.

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
- UnrealCV binds loopback (confirmed in §3.3: `env_ip = '127.0.0.1'`), and Windows does not
  filter loopback traffic, so no firewall approval is needed.

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

## 2. Windows: repo + conda env — DONE

The eval half of the repo has to exist on Windows too. Cloned under the user profile
(not `Documents`/`Desktop`, which OneDrive may sync):

```powershell
mkdir $env:USERPROFILE\UAV
cd $env:USERPROFILE\UAV
git clone https://github.com/cotictomaz/UAV-Flow.git
cd $env:USERPROFILE\UAV\UAV-Flow\UAV-Flow-Eval

# conda-forge only: sidesteps conda 26's Anaconda-channel ToS prompt entirely
conda create -n unrealcv python=3.11 -y -c conda-forge --override-channels
conda activate unrealcv
pip install -e .
```

Working root: `C:\Users\cotic\UAV\UAV-Flow\`.

The env is *named*, so it lives in `Miniconda3\envs\unrealcv` regardless of the directory
it was created from. Only `pip install -e .` cares about the working directory (it reads
`pyproject.toml`).

Installed versions: **gym 0.10.9, unrealcv 1.2.0, numpy 1.26.4** (downgraded from 2.4.6 —
see §3.6 triage). The 2018-era gym imports fine on Python 3.11 — no workaround needed for
`gym` itself.

**Two more environment gaps found at the §3.6 smoke test, not anticipated by this doc:**

1. **`pkg_resources` missing entirely.** `gym`'s lazy `import pkg_resources`
   (`registration.py:12`, hit inside `gym.make`) failed with `ModuleNotFoundError`. Cause:
   the env had `setuptools==84.0.0`, and setuptools stopped shipping `pkg_resources` as of
   ~v81 (it's slated for removal upstream by 2025-11-30). Fix:

   ```powershell
   pip install "setuptools<81"   # installed 80.10.2
   ```

   Importing `pkg_resources` afterward emits a `PkgResourcesDeprecationWarning` — expected,
   harmless, ignore it.

2. **numpy 2.x breaks `track.py:196`.** `get_tracker_init_point` does
   `direction = 2 * np.pi * np.random.sample(1)` (shape `(1,)`, not scalar) then
   `float(distance * np.cos(direction))`. numpy < 2 silently coerced a size-1 array through
   `float()`; numpy 2.x raises `TypeError: only 0-dimensional arrays can be converted to
   Python scalars`. This is exactly the risk flagged above — confirmed for real, not just
   theoretical. Fix:

   ```powershell
   pip install "numpy<2"   # installed 1.26.4
   ```

   `pip` will print a dependency-conflict warning (`opencv-python 5.0.0.93 requires
   numpy>=2`) — **ignore it**, verified harmless:

   ```powershell
   python -c "import cv2, numpy as np; a=np.zeros((4,4,3),dtype=np.uint8); print(cv2.cvtColor(a, cv2.COLOR_BGR2RGB).shape)"
   ```

   `cv2` still imports and runs fine at runtime; the constraint is opencv-python's build-time
   metadata, not an actual ABI break here.

**Both fixes are now pinned directly in `pyproject.toml`** (`numpy<2`, added `setuptools<81`
as an explicit dependency) — this is a real portability fix, not a local-path placeholder, so
unlike the edits in §9 it **is** committed. Re-verified end to end from a completely fresh
env to confirm this is what any new clone will now get automatically:

```powershell
conda remove --name unrealcv --all -y --override-channels -c conda-forge
conda create -n unrealcv python=3.11 -y -c conda-forge --override-channels
conda activate unrealcv
cd UAV-Flow-Eval
pip install -e .
```

Result: `setuptools` still initially lands at `84.0.0` (conda-forge's current default) but
the pin correctly downgrades it to `80.10.2`; `numpy` installs straight at `1.26.4`, and pip's
resolver backtracks `opencv-python` to `4.11.0.86` (a numpy-1.x-compatible build) instead of
the `5.0.0.93` used in the original manual fix — **no conflict warning this time**, a cleaner
resolution than the manual patch. `smoke_sim.py` (§3.6) then ran end-to-end with **zero manual
intervention**, printing a fresh camera config dict and exiting cleanly. Confirmed reproducible.

`pip install -e .` pulls `gym==0.10.9`, `unrealcv>=1.1.5`, `opencv-python`, `matplotlib`,
`simple_pid`, `pynput`, `docker`, `modelscope`, `numpy<2`, `setuptools<81` (from
`pyproject.toml`).

- [x] Verified — imports clean, no `collections.Iterable` breakage on 3.11:

```powershell
python -c "import gym, gym_unrealcv, unrealcv; print(gym.__version__, unrealcv.__version__)"
python -c "import numpy as np; from gym import spaces; print(spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32).sample())"
```

The second line exercises the continuous `spaces.Box` machinery `base_env.py` builds per
agent — the place numpy 2.x would most likely bite.

**Editable install matters:** settings are read from the repo working tree, not a copy —
`misc.get_settingpath` resolves `os.path.dirname(gym_unrealcv.__file__) + envs/setting/<file>`
(`gym_unrealcv/envs/utils/misc.py:14-17`). So editing the JSON in the repo is the right move.

## 3. Windows: simulator binary

### 3.1 Extract — DONE

Zip: `C:\Users\cotic\Downloads\Collection_WinNoEditor_0424_25.zip`, **47.7 GB**.

**Do not use `Expand-Archive`** at this size — it's a thin .NET wrapper that can take hours.
Windows' built-in `tar` (libarchive, handles ZIP64) is far faster:

```powershell
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\UAV\UE
cd $env:USERPROFILE\UAV\UE
tar -xf "$env:USERPROFILE\Downloads\Collection_WinNoEditor_0424_25.zip"
```

Extracting from inside the destination avoids `tar`'s `-C` path handling, which failed with
`could not chdir` here. Note `tar -C` also fails if the destination doesn't already exist —
`New-Item -Force` creates intermediate folders, plain `mkdir` of a nested path may not.

The zip **does** contain its own top-level folder, and there are **two** `Collection.exe`:

```
C:\Users\cotic\UAV\UE\Collection_WinNoEditor_0424_25\Collection.exe                          <- launcher shim, DO NOT USE
C:\Users\cotic\UAV\UE\Collection_WinNoEditor_0424_25\Collection\Binaries\Win64\Collection.exe <- the real one
```

- [x] `unrealcv.ini` confirmed present next to the real binary, with the exact layout
      `RunUnreal` assumes (see §3.3):

```ini
[UnrealCV.Core]
Port=9000
Width=640
Height=480
FOV=90.000000
EnableInput=True
EnableRightEye=False
```

`Width`/`Height` become `256` after the first launch — `ConfigUEWrapper` sets the resolution
and `write_resolution` rewrites the file. Backed up first:

```powershell
$ini = "$env:USERPROFILE\UAV\UE\Collection_WinNoEditor_0424_25\Collection\Binaries\Win64\unrealcv.ini"
Copy-Item $ini "$ini.bak"
```

Keep the zip until §3.6 passes; ~100 GB total until then.

### 3.2 Point the config at it — DONE

Edited `UAV-Flow-Eval/gym_unrealcv/envs/setting/Track/DowntownWest.json`, field `env_bin_win`
(it shipped with the authors' path, `D:\unrealzoo-gym-new\UnrealEnv\...`):

```json
"env_bin_win": "C:\\Users\\cotic\\UAV\\UE\\Collection_WinNoEditor_0424_25\\Collection\\Binaries\\Win64\\Collection.exe",
```

**Backslashes must be doubled** — it's JSON. Leave `env_bin` (Linux) and `env_map` alone;
`base_env.py:109-117` selects `env_bin_win` on `sys.platform == 'win32'`, and `env_map`
(`DowntownWest`) is what `set_map` switches to after launch.

- [x] Validated — parses, and the path resolves:

```powershell
python -c "import json,os; p=json.load(open(r'gym_unrealcv\envs\setting\Track\DowntownWest.json'))['env_bin_win']; print(p); print('exists:', os.path.exists(p))"
```

If this prints `exists: False`, dump the value with `repr()` before re-editing — the string
is valid JSON but wrong. Watch for the doubled `Collection` level
(`...\Collection_WinNoEditor_0424_25\Collection\Binaries\...`), which is easy to drop.

### 3.3 How launching actually works — VERIFIED (unrealcv 1.2.0)

**Do not start `Collection.exe` yourself before the eval script.** The env launches it:
`base_env.py:121` builds `RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)` and
`base_env.py:631-633` calls `ue_binary.start(...)`, which returns the `(ip, port)` that
`Character_API` then connects to. A manually-started instance won't be the one it talks to.

Read from `RunUnreal` source on this machine:

- **Absolute paths are used verbatim.** `__init__` branches on `os.path.isabs(ENV_BIN)`;
  absolute → `path2binary = os.path.abspath(ENV_BIN)`, no `UnrealEnv` joining. An
  `assert os.path.exists(self.path2binary)` fires immediately on a bad path — so a wrong
  `env_bin_win` fails loudly at `gym.make`, not mysteriously later.
- **The binary MUST be the one under `Binaries\Win64`, not the package-root shim.**
  `parse_path` does `part_path.index('Binaries')` — with the root-level `Collection.exe`
  that raises `ValueError`. This is a hard requirement, not a convention.
- **`path2env` is computed oddly on Windows** (`parse_path` prepends `/` to `C:`, giving
  `/C:\Users\...`) but it is only consumed by the Docker path and the commented-out
  `modify_permission`. Harmless for a local run.
- **Port**: `read_port()` parses `unrealcv.ini` as `int(ss[1][-4:])` — i.e. it assumes the
  second whitespace token is `Port=NNNN` with a 4-digit port. Default 9000.
- **`unrealcv.ini` is rewritten on every launch.** `write_resolution` overwrites lines 3
  and 4 by index (`Width=`, `Height=`) and `write_port` overwrites line 2. Both are
  line-index based and will corrupt an ini with a different layout. **Back it up first.**
- **IP is always `127.0.0.1`** (`local_host=True` default; base_env doesn't override).
  Confirms loopback → **no Windows Firewall involvement**, which matters here since there
  are no admin rights on this machine.
- **Port-free check relies on TCP self-connect.** On Windows `isPortFree` does
  `bind()` then `connect()` to the *same* address; it returns True only because a TCP
  simultaneous-open to yourself succeeds. If it ever misbehaves the symptom is a hang in
  the `while not isPortFree` loop, incrementing the port.
- **Rendering**: no `gpu_id` passed by `base_env.launch_ue_env`, `opengl=False` → **Vulkan**,
  `offscreen=False`, `nullrhi=False`. The map name is passed as the first CLI arg, so the
  binary boots straight into DowntownWest.
- **`start()` does not poll for readiness** — it is a flat `time.sleep(sleep_time)` and then
  it returns. See §3.4 below; this is the most likely first-run failure.
- **UE stdout/stderr go to `DEVNULL`**, so crashes are silent in the console. The real log
  is at `<package>\Collection\Saved\Logs\*.log`.

### 3.4 Raise the launch timeout before the first run — DONE

`base_env.py:631-633` passes `sleep_time=10`, and `RunUnreal.start()` treats that as the
*entire* startup budget — it sleeps, then hands the port to `Character_API` with no
readiness check. A 47.7 GB package booting DowntownWest from a cold file cache will not be
serving UnrealCV in 10 s, and the failure surfaces as a connection error that looks like a
networking problem rather than a timing one.

Raise it in `gym_unrealcv/envs/base_env.py:633`:

```python
nullrhi=self.nullrhi, sleep_time=50)   # was sleep_time=10 (actually set to 50, not 60)
```

Costs 40 extra seconds once per run (the binary launches on the first `reset()` only), and
removes the single most likely first-run failure. Can be tuned down once we know the real
cold-start time. **Confirmed sufficient** — §3.6 smoke test's `reset()` completed with room
to spare on both a cold and a warm shader cache.

### 3.5 Textures are *not* needed

`load_env.py -e Textures` is unnecessary here. `get_textures()` is only reached when
`'track_train' in env_name` (`augmentation.py:15-21`), and our env is `DowntownWest`.
The env id ends in `v0` → `reset_type == 0`, so `environment_augmentation` never fires either.

### 3.6 Smoke test — simulator alone, no server — DONE

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

Expected sequence:

1. A list of path components (`parse_path`'s debug `print`).
2. `Running docker-free env, pid:NNNN` — binary spawned.
3. `Please wait for a while to launch env......` then **60 s of silence** (§3.4). Not a hang.
4. Game window appears; first launch may sit compiling shaders and look frozen.
5. Two drones spawn, a camera config dict prints, the window closes.

- [x] Game window opens, loads DowntownWest, prints a camera config, exits cleanly. Actual
      output:
      ```
      {0: {'location': [-12179.343, -379.096, 18.211], 'rotation': [-0.0, -178.032, -0.0], 'fov': '90.000000'},
       1: {'location': [-11938.037, -310.679, 18.211], 'rotation': [-0.0, -179.77, -0.0], 'fov': '90.000000'}}
      ```
      Required the `pkg_resources`/setuptools and numpy fixes above — not a clean first pass.
- [x] Firewall: no prompt expected — §3.3 confirms `env_ip = '127.0.0.1'` and Windows does
      not filter loopback. Relevant because there are no admin rights here. Confirmed: no
      prompt appeared.

## 4. Networking: Linux server ← Windows client

### 4.0 What the "Linux server" actually is

Not a plain SSH-reachable host — it's a **Kubeflow notebook pod** on cluster `IVI-AWZ`,
reached via a kubeconfig ("user config file" for that cluster) plus the **Kubernetes
extension for VS Code**. No `sshd` runs on the pod, so a raw `ssh user@host` tunnel
(originally planned below) doesn't apply here. `kubectl port-forward` (or the same thing
via the VS Code extension's GUI) is the equivalent tool for a pod.

**Two things that are easy to conflate, worth stating explicitly:**

- **Where the server process runs** vs. **where you type the command to start it** are
  different. `python vla-scripts/openvla_act.py` must execute in a shell that's *inside the
  pod* — but it doesn't matter which client gives you that shell. A Jupyter terminal (in
  the notebook UI) and a VS Code Remote window (from either laptop) both just open a shell
  on the pod; the GPU work happens on the pod regardless of which one you used to type the
  command. Start it wherever is convenient.
- **The port-forward, by contrast, is local to whichever machine runs it.** It opens a
  listening socket on `127.0.0.1:<port>` only on that machine. Since `batch_run_act_all.py`
  runs on **this Windows box** and needs `127.0.0.1:5007` to resolve *here*, the
  port-forward must be set up **on this Windows machine specifically** — a forward set up
  from the other laptop (e.g. for its own VS Code Remote session) is a separate, unrelated
  tunnel and doesn't help this machine at all.
- **It's a live tunnel, not a persistent setting.** It only works while the forward is
  actively running (VS Code open, forward not stopped). If VS Code closes or the machine
  sleeps, it drops and must be re-started before the next eval run.

### 4.1 Set up the port-forward on Windows

1. Copy the kubeconfig ("user config file" for `IVI-AWZ`) from the other laptop to this
   machine, e.g. to `C:\Users\cotic\.kube\config` (check nothing's already there first;
   don't overwrite blindly). A USB drive is safer than email/cloud upload for a credential
   file like this.
2. Install the **Kubernetes** extension (Microsoft) in VS Code on this Windows machine, if
   not already present.
3. Open the Kubernetes panel in VS Code (sidebar icon) — it should list cluster `IVI-AWZ` →
   namespace → the pod also visible from the other laptop's session.
4. Right-click the pod → **Port Forward** (wording may vary by version) → add
   `5007:5007` (local:remote) **alongside** any existing forwards already listed for that
   pod (e.g. `8888:notebook-port`, `15090:http-envoy-prom` — Kubeflow's own Jupyter port and
   istio's metrics port; don't remove those). The remote side can be a raw port number even
   though it isn't a *named* container port like the other two — `5007` is just where the
   Flask process will bind once started, nothing in the pod spec needs to declare it.
5. Leave that VS Code window open for the entire eval session (§6).

- [ ] Port-forward `5007:5007` added and running on this Windows machine.
- [ ] Server started on the pod (§5) — from either laptop's terminal, doesn't matter which.
- [ ] `curl http://127.0.0.1:5007/predict` from Windows returns something other than a
      connection error (a 405/400 from Flask is expected and fine — a GET on a POST-only
      route — the point is only to rule out "nothing is listening").

### 4.2 Escape hatch, not currently needed

If port-forwarding is ever inconvenient, `batch_run_act_all.py` now takes a `--server_host`
flag (default `127.0.0.1`, so default behavior for anyone else using the script is
unchanged):

```powershell
python batch_run_act_all.py --server_host <reachable-host-or-ip>
```

Implemented directly (not left as a suggested edit) since it's a generic, backward-compatible
addition rather than a local path placeholder — safe to commit, unlike the edits in §9.

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
cd $env:USERPROFILE\UAV\UAV-Flow\UAV-Flow-Eval
conda activate unrealcv
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

- [x] Windows GPU + driver: RTX A4000 (8 GB), driver 580.92, CUDA 13.0
- [x] Env versions: Python 3.11 (conda-forge), gym 0.10.9, unrealcv 1.2.0, numpy 1.26.4,
      setuptools 80.10.2
- [ ] Checkpoint revision (HF commit hash) used for `model_path`
- [ ] `metric.txt` from the completed run

## 11. Progress

| Step | Status |
|---|---|
| §1 Windows prerequisites | **done** — no admin needed after all |
| §2 Repo + conda env | **done** — `C:\Users\cotic\UAV\UAV-Flow`, env `unrealcv` |
| §3.1 Extract simulator | **done** — 47.7 GB via `tar` |
| §3.2 Configure `env_bin_win` | **done** — validated `exists: True` |
| §3.3 Launcher behavior | **done** — read from source, findings recorded |
| §3.4 `sleep_time` 10 → 60 | **done** |
| §3.5 Textures | **n/a** — not needed for DowntownWest |
| §3.6 Simulator smoke test | **done** — required `setuptools<81` and `numpy<2` fixes, see §2 |
| §4 Kubeflow port-forward (was: SSH tunnel) | **in progress** — plan settled (§4.0–4.1), pod
identified as Kubeflow notebook on cluster `IVI-AWZ`; user setting up kubeconfig + VS Code
Kubernetes extension port-forward next |
| §5 Linux inference server | not started |
| §6 Evaluation run | not started |

### Local edits made (do not commit — see §9)

| File | Change |
|---|---|
| `gym_unrealcv/envs/setting/Track/DowntownWest.json` | `env_bin_win` → local extraction path |
| `gym_unrealcv/envs/base_env.py:633` | `sleep_time=10` → `sleep_time=50` |
| `<package>\...\Win64\unrealcv.ini` | backed up to `unrealcv.ini.bak`; rewritten by launcher |
| `UAV-Flow-Eval/smoke_sim.py` | new scratch file for §3.6, not part of the repo |
| `unrealcv` conda env: `setuptools` | `84.0.0` → `<81` (`80.10.2`) — restores `pkg_resources` |
| `unrealcv` conda env: `numpy` | `2.4.6` → `<2` (`1.26.4`) — fixes `track.py:196` `TypeError` |

### Committed (not a local-only edit — see §4.2)

| File | Change |
|---|---|
| `UAV-Flow-Eval/batch_run_act_all.py` | added `--server_host` flag (default `127.0.0.1`); `server_url` now built from it instead of a hardcoded `127.0.0.1` |