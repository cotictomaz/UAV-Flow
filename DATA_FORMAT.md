# DATA.md

Source of truth for the UAV-Flow data format: what a sample is, what the fields mean, and what coordinates and order actions are expressed in. Covers **both** datasets — UAV-Flow (real) and UAV-Flow-Sim.

Provenance tags on every claim:

- **[measured]** — verified empirically. Real: `/home/jovyan/voxel51-vol-1/UAV-Flow`, shards `train-0000{0,1,2}-of-00054`, **1500 trajectories / 100,861 frames** (a ~6% sample). Sim: `/home/jovyan/voxel51-vol-1/UAV-Flow-Sim`, **all 21 shards — the complete dataset, 10,109 trajectories / 312,240 frames**.
- **[paper]** — from *UAV-Flow Colosseo* (arXiv:2505.15725v2); not verifiable from these shards.
- **[code]** — a property of this repo's loader, not of the data.

**Sim statistics are exact — the whole dataset, not a sample.** The measured episode count, 10,109, matches the paper's figure exactly. Real remains a 1500-trajectory sample of 54 shards: its format claims held 1500/1500 and should generalize, but its distribution statistics are still estimates.

> **Sim numbers here supersede an earlier 3-shard sample, and several moved a lot.** That sample under-drew one of sim's two collection regimes (§1.1), so it understated the action ranges by ~1.7× (dx q99 29.2 → **50.7**), overstated the 10/20 cm discretisation (62% → **52%**), and put the primitive-instruction share at 57.2% when it is **41.7%**. Three claims are now outright **corrected**: the sim train/test dates do *not* separate (§7.8), preprocessed reconstruction has a **15.5 m** outlier tail rather than being exact (§4), and dyaw's ±3.00° cap has 7 violations (§7.5). If you have a checkpoint whose `norm_stats` came from that sample, its scale is wrong. Conclusions that **survived**: units, every coordinate convention, one map, zero train/test id overlap.

Where paper and data disagree, the data wins. See [Known quirks](#7-known-quirks) — several are load-bearing.

---

## 1. The two datasets, the unit trap, and sim's two regimes

| | UAV-Flow (real) | UAV-Flow-Sim |
|---|---|---|
| Source | DJI Mavic 3T RTK, 3 campuses **[paper]** | UnrealCV campus map **[paper]** |
| **Position units** | **metres** **[measured]** | **Unreal centimetres** **[measured]** |
| Rotation units | degrees **[measured]** | degrees **[measured]** |
| `raw_logs` row width | **7** (trailing timestamp) **[measured]** | **6** (no timestamp), 10,109/10,109 **[measured]** |
| Raw yaw range | `[-180, 180]` **[measured]** | **`[-180, 360]`** (mixed — two conventions, §7.3) **[measured]** |
| Shards | 54 | 21 (20 × 500 traj + a 109-traj tail) **[measured]** |
| Episodes | ~30K **[paper]**; 500/shard × 54 ⇒ ~27K **[measured]** | **10,109 exactly** — full count, matches paper **[measured]** |
| Frames | 100,861 (sample) **[measured]** | **312,240** (all) **[measured]** |
| Frames/trajectory (median) | 61 | **25** |
| Test set | — | 273 trajectories (`UAV-Flow-Eval/test_jsons/`) **[measured]** |
| On-disk | — | 35.9 GB **[measured]** |

> ### ⚠️ The 100× trap
>
> **Real is metres. Sim is centimetres.** Both now measured, not inferred:
>
> - **Sim**: **3,754** sim instructions state an explicit distance ("Move 10.0 meters…"). Stored net displacement ÷ stated metres = **exactly 100.00** (median, p25 and p75 all 100.00, across all 3,754). "Move 10.0 meters" ⇒ net displacement of exactly 1000.0 stored units. **[measured]**
> - **Real**: reconstructed local displacements match instruction semantics at ~0.13 per step, i.e. 0.66 m/s — metres. **[measured]**
> - `metric.py` divides sim positions by 100 to score in metres. **[code]**
>
> **Why it bites:** `unnorm_key` is the hardcoded string `"sim"` on both the training and serving sides regardless of which dataset was used (see CLAUDE.md). A checkpoint carries whatever `norm_stats` its training data implied, so a real-trained policy un-normalizes to **metres** and the simulator executes them as **centimetres** — ~100× too small. Nothing errors.
>
> **Concretely:** `batch_run_act_all.py` treats a step below `ACTION_SMALL_DELTA_POS = 3.0` (cm) as "stalled" and aborts after 10 such steps. A real-trained policy emits ~0.13 (metres) → 0.13 < 3.0 → **every episode aborts after ~10 steps as a false stall.** **[code]**
>
> Train on sim to evaluate in sim, or rescale by 100.

Note the mismatch is **not** a clean 100× in `norm_stats` terms, because the two action distributions differ in shape as well as scale — and **`dyaw` is in radians in both, so it is not unit-scaled at all** **[measured]**:

| `norm_stats["sim"]` written by `finetune_uav.py` | dx | dy | dz | dyaw |
|---|---|---|---|---|
| **real** q01 → min (1500-traj sample) | −0.2505 | −0.2350 | −0.0885 | −0.1360 |
| **real** q99 → max (1500-traj sample) | 0.5802 | 0.2492 | 0.0931 | 0.0980 |
| **sim** q01 → min (**all 10,109**) | **−37.0200** | **−35.4884** | −20.0 | −0.0524 |
| **sim** q99 → max (**all 10,109**) | **50.6561** | **33.9309** | 20.0 | 0.0524 |
| ratio sim/real (q99) | 87.3× | 136.2× | 214.8× | **0.53×** |

These are the exact values `finetune_uav.py` will write when trained on the full sim dataset — the dx/dy ranges are **~1.7× wider** than the earlier 3-shard sample implied, because that sample under-represented the fast object-interactive regime (§1.1).

### 1.1 — Sim is two collection regimes, split by date ⚠️

**[measured]** This is invisible in a small sample and it governs what the data can teach. The 10,109 trajectories fall into two disjoint, internally-consistent regimes, separated cleanly by the date in the trajectory id:

| | **2025-03-27** — *primitive / discretised* | **6 other dates** — *object-interactive / continuous* |
|---|---|---|
| Trajectories | 5,003 (49.5%) | 5,106 (50.5%) |
| Frames | 170,678 (54.7%) | 141,562 (45.3%) |
| Frames/traj min/med/mean/max | 4 / 21 / 34.1 / 379 | 7 / 27 / 27.7 / 115 |
| Instructions | **84.3% state explicit metres/degrees** ("Rotate 75 degrees to the left") | **0% metric** — all object-referential ("Pass by the car from the left side") |
| Motion | **99.4%** of nonzero steps are exactly **10.00 or 20.00 cm** | **0%** discretised; median step **31.38 cm**, continuous |
| dx q01→q99 | −10.0 → 20.0 | **−40.75 → 52.98** |
| dyaw q01→q99 | −0.0524 → 0.0524 (**±3.00°, hard cap**) | **−0.0005 → 0.0005 (≈ frozen)** |
| Rotation | **1,276 of the 1,295** trajectories that rotate >5° | 19 |
| Turn/rotate/face instructions | **1,205** | **0** |
| Climb/ascend instructions | **405** (all of them) | **0** |
| Raw yaw convention | **`[0, 360]`** — 50.4% of frames >180° | `[-180, 180]` — 0% >180° |
| Roll/pitch > 1° | 280 (5.6%) | 21 (0.4%) |
| Position steps exactly 0 | 14.1% | 3.6% |

Dates: 03-27 is the primitive regime; 03-14, 04-30, 05-01, 05-02, 05-07, 05-08 are the object regime.

**Why it matters:**

- **All yaw and all climb signal lives in 2025-03-27.** The object half is effectively yaw-frozen (99% of its steps rotate <0.03°) and contains no climb instruction at all. Train on the object half alone and the policy cannot learn to turn or ascend — the Turn/Rotate/Ascend classes `metric.py` scores would be unlearnable.
- **The two halves disagree about scale.** The pooled `norm_stats` above are a blend of a 20 cm-quantised regime and a ~31 cm continuous one, so they fit neither. The pooled dyaw range is set entirely by 03-27's ±3° cap; the object half occupies 1% of it.
- **The `[0, 360]` yaw quirk (§7.3) is purely a 03-27 artifact** — 2,746 of the 2,746 affected trajectories are from that date.
- Both regimes are ~half the data, so a random split mixes them. If you filter by date, know what you are dropping.

---

## 2. Physical layout

Both datasets: **exactly 500 trajectories per shard**, identical schema. Sim's last shard (`train-00020`) holds the remaining **109** — 20 × 500 + 109 = 10,109. **[measured]**

Verified invariants — **10,109/10,109 in sim (all of it)** and 1500/1500 in the real sample **[measured]**:

- Every trajectory is **fully contained in one shard**. (`prepare_data.py` buffers frames until a trajectory is complete; it would never flush a split trajectory.)
- A trajectory's rows are **contiguous and in ascending `frame_idx`** order.
- `frame_idx` is **contiguous `0..N-1`**, no gaps.
- `len(raw_logs) == len(preprocessed_logs) == number of image rows`.

## 3. Parquet schema — one row = **one frame** (identical in both)

```
id         string                              # trajectory id, repeated on every frame
frame_idx  int32                               # 0..N-1 within the trajectory
image      struct<bytes: binary, path: string> # `path` is always null; `bytes` is a PNG
log        string                              # JSON: the WHOLE trajectory, repeated on every frame
```

- `image.bytes`: **256×256 RGB PNG** in both datasets; `image.path` always null. Sampled across all 21 sim shards — 126/126 were 256×256 RGB PNG. **[measured]**
- `id`: `YYYY-MM-DD_HH-MM-SS` in both. For **real** it is the flight start in **UTC+8** (equals `raw_logs[0][6] + 8h` to within ~1 s) **[measured]**. For **sim** there is no timestamp column to cross-check against — but the date component is *not* cosmetic: it is what separates the two regimes (§1.1).
- `log`: **identical on every row of a trajectory** — 89× redundancy for real (777 MB stored vs 8.7 MB unique per shard); for sim, **78.9×** across the whole dataset (**3,444 MB of decompressed `log` strings vs 43.6 MB unique**; ~164 MB vs 2.08 MB per shard). **Parse it once per `id`, not per row.** **[measured]**

  (The redundancy factor exceeds the mean trajectory length of 30.9 because it is length-weighted — long trajectories carry both more rows *and* a longer `log`, so the ratio tracks E[N²]/E[N].)

## 4. The `log` payload

Exactly four keys, on **all 10,109 sim trajectories** and all 1500 sampled real ones **[measured]**:

```jsonc
{
  "raw_logs":          [[x, y, z, roll, yaw, pitch, timestamp], ...],  // real: N × 7
                       [[x, y, z, roll, yaw, pitch], ...],             // sim:  N × 6
  "preprocessed_logs": [[x, y, z, roll, yaw, pitch], ...],             // both: N × 6
  "instruction":       "Make way to the road sign from the left side",
  "instruction_unified": "Go toward the road sign from the left side"
}
```

`prepare_data.py` adds `length` and `id` when writing the folder format; the parquet has neither. **[code]**

### `raw_logs`

Sim column below is now the **full-dataset** range. **[measured]**

| idx | field | real (metres, sample) | sim (centimetres, all 10,109) |
|---|---|---|---|
| 0 | x | 3,309,261 … 3,363,144 | −29,233 … 16,715 |
| 1 | y | 213,851 … 786,013 | −8,467 … 31,766 |
| 2 | z | −9.9 … 23.3 | **−172** … 1,574 |
| 3 | roll (deg) | −25.6 … 28.4 | −56.3 … 55.5 (**≈0**, std 0.41; 63.0% of frames \|v\|<0.01) |
| **4** | **yaw** (deg) | −179.9 … 179.9 | **−180 … 360** |
| 5 | pitch (deg) | −30.5 … 33.7 | −134.5 … 151.1 (**≈0**, std 0.89; 82.2% \|v\|<0.01) |
| 6 | timestamp | Unix sec, float | **absent** |

**The rotation order is `[roll, yaw, pitch]` — yaw is index 4, not 3.** Confirmed independently in both datasets: only index 4 has a wide angular range (540° sim / 360° real) **[measured]**; index 4 reconstructs the heading of travel in the real data **[measured]**; and the paper's nDTW definition says "position (x, y, z) with the cosine values of the orientation (roll, yaw, pitch)" **[paper]**. (Prose elsewhere in the paper says "(roll, pitch, yaw)" — that is wrong; ignore it.) `gym_unrealcv` uses the same `[roll, yaw, pitch]` order.

Index 6 is the **7th field the codebase silently ignores**, and its absence in sim is why the same code works on both: every consumer slices `[:, [0,1,2,4]]`, valid for width 6 and 7 alike. Sim `test_jsons` are likewise 6-wide.

**Real: the global frame is opaque.** x/y are a large planar projection in metres (northing ≈3.3e6 ⇒ ~30°N, consistent with UTC+8), not raw lat/lon. **Treat it as an arbitrary site-local metre grid** — it is only ever used via differences from frame 0. Greedy 3-km clustering of start positions yields **exactly 3 sites** **[measured]**, matching the paper's three campuses spanning 5.02 km²:

| site | trajectories | extent |
|---|---|---|
| 0 | 723 | 1.56 × 1.71 km |
| 1 | 240 | 0.71 × 0.88 km |
| 2 | 537 | 0.87 × 0.57 km |

**Sim: one map.** Greedy 300-m clustering of start positions yields **2 clusters** — but that is a threshold artifact, not a second site: 10,090 trajectories in the main cluster and **19** in a pocket whose centroid sits **302.5 m** away, just over the cutoff. Treat it as **one map**. **[measured]**

- Start positions span **452.3 × 376.0 m**, altitude **0.69 … 11.00 m**.
- All frames span x −29,233 … 16,715 cm and y −8,467 … 31,766 cm (≈459 × 402 m).

Sim roll/pitch are near-always zero (63% / 82% of frames have |v| < 0.01°), and only **301 / 10,109 (3.0%)** trajectories have any |roll| or |pitch| > 1°. Those are synthetic primitives like *"Climb 3.0 meters with a 35-degree angle"*, they are **overwhelmingly from the 2025-03-27 regime** (280 of 301, §1.1), and their extremes are large — up to 56° roll and 151° pitch. They also drive the reconstruction tail below. **[measured]**

### `preprocessed_logs` — N × 6 (both)

The same pose **re-expressed in the frame of the trajectory's first frame**. Row 0 is exactly all-zeros in all 10,109 sim trajectories and all 1500 sampled real ones. **[measured]** Paper: "we transform global GPS coordinates into a local Cartesian coordinate system centered at the trajectory's starting position. Relative orientation is also computed with respect to the initial frame." **[paper]**

Reconstruction from `raw_logs` **[measured]**:

```python
yaw0 = np.deg2rad(raw[0, 4])
R = np.array([[np.cos(-yaw0), -np.sin(-yaw0), 0],
              [np.sin(-yaw0),  np.cos(-yaw0), 0],
              [0, 0, 1]])
pre[:, 0:3] ?= (raw[:, 0:3] - raw[0, 0:3]) @ R.T
pre[:, 3:6] ?= wrap180(raw[:, 3:6] - raw[0, 3:6])
```

| | real (sample) | sim (all 10,109) |
|---|---|---|
| XY error | median 0.005 m, p95 0.074 m ✅ | median **0.000000**, p95 0.12 cm ✅ |
| **Z error** | median 0.181 m, **max 6.78 m** ❌ **§7.1** | median **0.000000**, p95 1.79 cm ✅ |
| rotation error | **exactly 0.000000°** ✅ | **exactly 0.000000°** — max 0.0 across all 312,240 frames ✅ |

The rotation is **yaw-only** — roll and pitch do not tilt the frame. Rotation columns are plain per-frame deltas from frame 0, wrap-corrected.

**Sim reconstructs essentially exactly for the bulk** — 29% of frames bit-exact, **89% within 1 cm, 98% within 10 cm** (smaller than one 20 cm step). Rotation is exact everywhere.

But the full dataset exposes a **long tail the 3-shard sample missed**: **265 trajectories (2.6%) exceed 10 cm** and **71 (0.7%) exceed 1 m**, with a worst case of **15.5 m**. The tail is squarely the roll/pitch cohort — 197 of the 265 worst have |roll| or |pitch| > 1°, and 197 of those 301 trajectories land in the tail (correlation with the roll/pitch flag r=0.44). Every one of the worst eight is a 2025-03-27 pitched primitive: **[measured]**

```
2025-03-27_20-39-22  "Ascend 10.0 meters at 50 degrees"          err 15.5 m  (max roll/pitch 105°)
2025-03-27_17-22-01  "Move 8.0 meters to the 10-degree right…"   err 12.0 m  (max roll/pitch 122°)
2025-03-27_18-00-22  "Climb to an altitude of 7.0 meters…"       err  9.7 m  (max roll/pitch  99°)
```

So the earlier "sim reconstructs exactly, full stop" claim is right for ~97% of trajectories and **wrong for a pitched-primitive tail**. The yaw-only reconstruction above cannot represent a frame that is genuinely rolled or pitched, which is exactly where it breaks. Unlike real §7.1 this is not a drift artifact — it is a modelling limit of the yaw-only transform, and it is confined to trajectories the sim itself flew at extreme attitude.

**Real does not**, on the z axis — see §7.1. This is the single biggest difference between the two datasets' trajectory quality.

## 5. Coordinate conventions — **identical in both datasets**

Established from instruction semantics, the decisive test. Verified **independently** in real and sim; every convention agrees, and the full sim dataset confirms each one on 4–30× more trajectories than the earlier sample. **[measured]**

| axis | direction | real evidence (sample) | sim evidence (**all 10,109**) |
|---|---|---|---|
| **+x** | **forward** | "approach/toward" → **+5.58 m** (93% pos, n=428); "back off" → **−5.12 m** (2% pos, n=58) | "forward/advance" → **+632.7 cm** (96% pos, n=505); "approach/toward" → **+471.2 cm** (82% pos, n=1119); "back off/retreat" → **−446.9 cm** (**0% pos**, n=341) |
| **+y** | **right** | "shift right" → **+4.55 m** (100% pos, n=32); "shift left" → **−2.99 m** (0% pos, n=22) | "right/starboard" → **+134.9 cm** (76% pos, n=2189); "left/port" → **−120.1 cm** (11% pos, n=2673) |
| **+z** | **up** | "descend/land" → **−0.75 m** (6% pos, n=16) | "climb/ascend" → **+375.9 cm** (**99% pos**, n=405); "descend/land" → **−236.9 cm** (**0.9% pos**, n=851) |
| **+yaw** | **clockwise** (right turn, from above) | "clockwise" → **+363.6°** (92% pos, n=50); "counterclockwise" → **−368.6°** (1% pos, n=105) | "Turn/rotate … right" → **+75.0°** (82% pos, n=188); "… left/counterclockwise" → **−75.0°** (**0% pos**, n=201) |

The sharpest sim evidence comes from the 2025-03-27 metric primitives, which state the answer outright and match it exactly **[measured]**:

```
"Turn right by 165 degrees"      -> net unwrapped yaw +165.0°
"Rotate 75 degrees to the left"  -> net unwrapped yaw  −75.0°
"N degrees to the right" (n=985)  -> net y +229.8 cm (88% pos)
"N degrees to the left"  (n=1028) -> net y −200.0 cm ( 0% pos)
```

An independent check confirms handedness without relying on wording at all: for the 224 sim trajectories that both translate and turn, the **unwrapped heading of travel `atan2(dy, dx)` tracks raw yaw with correlation > 0.5 in 224/224** — heading and yaw increase together, so +yaw rotates from +x (forward) toward +y (right), i.e. clockwise seen from above. **[measured]**

This is **x-forward, y-right, z-up with yaw positive clockwise** — a left-handed frame, i.e. the **Unreal Engine convention**, consistent across both halves of the project.

> **Measure rotation with `np.unwrap`.** Net *wrapped* yaw for a full-circle instruction is ≈0°, making clockwise and counterclockwise indistinguishable (real: frac>0 of 0.48 vs 0.39 — pure noise). Unwrapped, they separate cleanly at ±360°. 29 real trajectories are full-circle spins. **[measured]**

The sim's vertical axis is far cleaner than the real one: 99% of "climb" instructions move up, versus a real z channel that is largely sensor-limited (§7.2). But note **all 405 climb/ascend instructions live in the 2025-03-27 regime** (§1.1) — the object-interactive half of sim never climbs.

## 6. The action space

**Actions are not stored. They are derived** by `SimpleVLADataset._process_episode` (`prismatic/vla/datasets/uav_dataset.py`). **[code]**

An action is the **next pose expressed in the current frame's local frame** — a one-step relative delta, *not* a velocity and *not* an absolute waypoint:

```python
traj_raw = raw[:, [0, 1, 2, 4]]          # x, y, z, yaw   <- from RAW logs (works for width 6 and 7)
traj_raw[:, 3] = np.deg2rad(traj_raw[:, 3])
# action[i] = pose[i+1] expressed in the local frame of pose[i]
action[i] = [ local_dx, local_dy, local_dz, wrap_pi(yaw[i+1] - yaw[i]) ]
action[-1] = [0, 0, 0, 0]                # last action forced to zero (the "stop" signal)
```

### Action vector: `[dx, dy, dz, dyaw]`, length 4

| idx | field | frame | units |
|---|---|---|---|
| 0 | dx | forward, current body frame | **m** (real) / **cm** (sim) |
| 1 | dy | right, current body frame | **m** (real) / **cm** (sim) |
| 2 | dz | up (world-vertical; the rotation is yaw-only) | **m** (real) / **cm** (sim) |
| 3 | dyaw | — | **radians** in both, wrapped to [−π, π) |

**Mixed units within one vector**: positions in metres/centimetres, yaw in **radians** — while proprio yaw stays in **degrees**. `uav_dataset.py:142` converts yaw only on the copy used to derive actions, deliberately leaving proprio in degrees.

### Measured action distributions **[measured]**

**Real** — 100,861 steps, positions in **metres**:

| | mean | std | min | q01 | q99 | max |
|---|---|---|---|---|---|---|
| dx (m) | 0.1075 | 0.1534 | −1.6081 | −0.2505 | 0.5802 | 2.6305 |
| dy (m) | 0.0087 | 0.0874 | −1.0085 | −0.2350 | 0.2492 | 1.3423 |
| **dz (m)** | 0.0003 | 0.0212 | −1.2310 | −0.0885 | 0.0931 | 0.7213 |
| dyaw (rad) | −0.0053 | 0.1114 | −3.1391 | −0.1360 | 0.0980 | 3.1341 |

Median step 0.132 m ⇒ **≈0.66 m/s** at 5 Hz. dyaw q01/q99 ≈ −7.8°/+5.6° per step.

**Sim** — **312,240 steps (the whole dataset)**, positions in **centimetres**. One action per frame: the last of each episode is the forced zero, so steps == frames.

| | mean | std | min | q01 | q99 | max |
|---|---|---|---|---|---|---|
| dx (cm) | 11.1561 | 18.5357 | −63.1639 | −37.0200 | 50.6561 | 72.7579 |
| dy (cm) | −0.1026 | 12.3613 | −63.1648 | −35.4884 | 33.9309 | 58.0980 |
| dz (cm) | −0.1190 | 5.2046 | −46.2810 | −20.0000 | 20.0000 | 30.6210 |
| dyaw (rad) | −0.0002 | **0.0155** | −0.2754 | −0.0524 | 0.0524 | 0.3924 |

**These are pooled across two regimes that disagree (§1.1), so the pooled shape is bimodal and describes neither half well.** Per regime:

| | 03-27 (primitive) | other 6 dates (object) |
|---|---|---|
| dx q01 → q99 | −10.0 → 20.0 | −40.75 → 52.98 |
| dy q01 → q99 | −19.92 → 19.92 | −39.57 → 38.02 |
| dz q01 → q99 | −20.0 → 20.0 | −23.60 → **1.20** |
| dyaw q01 → q99 (rad) | −0.0524 → 0.0524 | **−0.0005 → 0.0005** |
| median nonzero step | **20.00 cm** | **31.38 cm** |

Median nonzero step overall is **20.00 cm** ⇒ 1.0 m/s if the sim ran at 5 Hz (unverifiable — no timestamps).

**Discretisation is regime-specific, not a property of sim as a whole.** Overall **51.5%** of nonzero steps are exactly 10.00 or 20.00 cm (69,691 and 76,105 steps) — but that is **99.4% within 2025-03-27 and 0.0% everywhere else**. The object regime is fully continuous. Likewise **9.4% of steps have exactly zero position** (3.2% are zero in all four components), concentrated in 03-27 (14.1% vs 3.6%). **[measured]**

**dyaw is capped at ±3.00°/step (±0.0524 rad) — in the 03-27 regime, where its max is exactly 3.00°.** The cap is *not* a global invariant: **7 steps across the object regime exceed it**, reaching **22.48°** (§7.5). Both q01/q99 sit exactly at ±3.00° because 03-27 supplies essentially all the yaw motion. **[measured]**

Normalization **[code]**: actions scale to [−1, 1] against the **1st/99th percentile** (not min/max), then clip — **2.0% of real** and **1.6% of sim** action components are clipped. These percentiles become the checkpoint's `norm_stats["sim"]` (`q01→min`, `q99→max`).

### State / proprio

Proprio comes from **`preprocessed_logs`** (not raw): `[x, y, z, yaw]` = columns `[0,1,2,4]`, in the **first-frame** frame, in **metres (real) / centimetres (sim)** and **degrees**. It is formatted into the prompt as text, `round(x, 1)` comma-joined:

```
In: Current State: 3.7,0.0,-0.1,-0.8, What action should the uav take to {instruction}?
Out:
```

Note the asymmetry, easy to get wrong: **proprio is in the first-frame frame (from `preprocessed_logs`); actions are in the current-frame frame (derived from `raw_logs`).** Different sources, different frames.

| proprio range | real (m / deg, sample) | sim (cm / deg, **all 10,109**) |
|---|---|---|
| x | −28.8 … 44.7 | −995.5 … 2407.5 |
| y | −13.8 … 14.4 | −1064.2 … 1530.3 |
| z | −7.08 … 3.60 | −1000.0 … 1000.0 |
| yaw | −180 … 180 | −179.93 … 180.0 |

## 7. Known quirks

Mostly unflagged upstream. §7.1 and §7.2 are real-only and both concern the vertical axis; §7.3 differs between datasets.

### 7.1 — REAL ONLY — `preprocessed_logs[:, 2]` (z) is not the raw altitude; it carries a distance-proportional drift ⚠️

Columns 0, 1, 3, 4, 5 reconstruct from `raw_logs` essentially exactly. **Column 2 does not**: median error 0.181 m, p95 1.41 m, **max 6.78 m**. **[measured]** The equivalent sim error is **exactly zero** — this is a real-data-only defect.

It is not noise — preprocessed z is a near-perfect **linear ramp in horizontal distance travelled**. Of 950 real trajectories with meaningful motion, **70% have |corr(pre_z, horizontal distance)| > 0.9**; median slope −0.031 (~3% downward grade). Worst case:

```
traj 2025-04-15_14-51-21  "Go through the tree from the right side"
  raw z-z0 :  0.00 → −0.30 m   (essentially level flight)
  pre z    :  0.00 → −7.08 m   (corr with distance = −0.9991, fit −0.181·dist, resid 0.20 m)
```

A drone does not descend 7 m while its own altimeter reports 0.3 m over a 37 m horizontal pass. **The ramp is an artifact.** Its angle correlates with initial pitch (r=0.51, fit slope 0.60) but not cleanly enough to call the mechanism — **I could not identify the cause and do not claim one.**

**Guidance:** `raw_logs` z is the trustworthy real altitude (coarse — see §7.2). Do not treat real preprocessed z as ground-truth altitude, and note that **proprio z is fed to the model from this drifting signal**, with error growing over flight length.

### 7.2 — REAL ONLY — Altitude is stair-stepped; the `dz` action is exactly zero 90% of the time ⚠️

Real `raw_logs` z is sample-and-held at ~0.1 m resolution while x/y update every frame. **[measured]**

- Consecutive raw z **exactly equal: 90.0%** of steps (vs 0.6% for x).
- **618 / 1500 trajectories (41%) — raw z never changes at all.**
- ⇒ **90.15% of derived real `dz` actions are exactly 0.0**; the dz normalization range is ±~0.09 m.

**Sim looks superficially similar but is not the same problem.** Sim dz is exactly zero on **82.0%** of steps and **7,632/10,109 (75%)** of sim trajectories never change altitude — *more* trajectories than real. But sim z is **exact**: a zero means the drone genuinely held altitude, and when it does move it moves a clean ±20 cm, with 99% of "climb" instructions going up. In the real data a zero usually means **the sensor had not updated**. Same statistic, opposite meaning.

**The two sim regimes fail differently on the vertical axis, though** (§1.1) **[measured]**:

| | 03-27 (primitive) | other 6 dates (object) |
|---|---|---|
| dz exactly 0 | 86.4% | 75.3% |
| sub-0.5 cm jitter | **0.0%** | **16.2%** |
| meaningful moves (\|dz\| > 0.5 cm) | 13.6% | 8.5% |
| …of those, fraction **upward** | **56.8%** | **19.2%** |
| climb/ascend instructions | 405 | **0** |

So the object regime's z channel is not clean-but-idle — it is **jittery and almost purely downward**: 16% of its steps are sub-millimetre noise rather than true holds, and its q99 for dz is just **+1.2 cm** because it essentially never climbs. Its 479 descend/land instructions have no ascending counterpart. **All genuine, clean vertical signal in the sim comes from 2025-03-27.**

Consequence: only 4.9% of real steps have |dz| > 0.05 m, and only 6 real instructions mention ascend/descend. **Do not expect meaningful learned altitude control from the real data alone** — the sim is where the vertical signal lives, and within sim it lives in one date.

### 7.3 — Raw yaw needs explicit wrapping — and the two datasets wrap differently

| | real (sample) | sim (**all 10,109**) |
|---|---|---|
| raw yaw range | `[-180, 180]` | **`[-180, 360]`** — mixed |
| frames above 180° | 0 | **86,052 (27.6%)** |
| trajectories affected | — | **2,746 / 10,109** |
| trajectories where naive differencing breaks | **179 / 1500 (12%)** | 341 / 10,109 (3.4%) |

**[measured]** Sim raw yaw is **not consistently normalized**: 27.6% of frames lie in (180, 360]. `preprocessed_logs` is correctly wrapped in both (range −179.93 … 180), and `uav_dataset.py` applies `(d + π) % 2π − π`, which handles both conventions because it operates on *differences*. **Any new code differencing `raw_logs[:, 4]` must wrap explicitly.** Real example: `2025-04-24_11-48-52` frame 34, raw yaw −179.0 → 179.5 — a naive delta of +358.5° that should be −1.5°.

**The mixed convention is entirely a 2025-03-27 artifact** (§1.1) — the two regimes each use *one* convention, consistently **[measured]**:

| date | raw yaw range | frames >180° | traj >180° |
|---|---|---|---|
| **2025-03-27** | **`[0, 360]`** | **50.4%** | **2,746 / 5,003** |
| 2025-03-14 | `[-180, 180]` | 0% | 0 / 484 |
| 2025-04-30 … 2025-05-08 | `[-180, 180]` | 0% | 0 / 4,622 |

So sim is not "inconsistently normalized" within a session — it is two sessions with different conventions, concatenated. 303 of the 341 naive-differencing breaks are from 03-27; the remaining 38 are ordinary ±180° wrap crossings.

### 7.4 — REAL ONLY — Five trajectories have an empty `instruction_unified`

`2025-04-22_08-36-58`, `2025-04-20_15-54-42`, `2025-04-20_16-00-39`, `2025-04-22_08-29-52`, `2025-04-21_13-31-13` — all "…toward the front of the camera" instructions, with `instruction` populated but `instruction_unified` = `""`. **[measured]** **No sim trajectory has an empty `instruction` or `instruction_unified` — now confirmed on all 10,109, not just a sample.**

Currently harmless: training always reads `instruction` **[code]**, and `batch_run_act_all.py` falls back to `instruction`. It becomes a silent bug the moment anything trains on `instruction_unified` — those five train on an empty prompt. Filter explicitly.

### 7.5 — Per-step yaw glitches — severe in real, vanishingly rare in sim

170 real steps (0.17%) have |dyaw| > 90° in a single 0.2 s interval, up to 179.86° — physically implausible at 5 Hz and not explained by wrapping; 290 steps (0.29%) exceed 30°. These are tracking glitches and land directly in the action distribution (the ±3.13 rad tails in §6). Consider filtering the real data. **[measured]**

**Sim still has zero steps above 30°**, so the real-vs-sim contrast holds — but the full dataset weakens "hard-capped at ±3.00°/step" to *almost* always **[measured]**:

- **7 steps of 312,240 (0.0022%) exceed 3.001°**, across **7 trajectories**, reaching **22.48°**.
- All 7 are in the **object regime** (§1.1); within 2025-03-27 the max is **exactly 3.00°** and the cap is exact.
- 4 steps exceed 5°, 2 exceed 10°, 1 exceeds 20°.

Examples: `2025-04-30_20-25-54` ("Suggest proceeding to the tree on the front side", 7.71°), `2025-03-14_16-44-06` (3.16°). Negligible for statistics — 0.002% — but do not rely on ±3.00° as an invariant when validating a policy's output or writing an assertion.

### 7.6 — `prepare_data.py` re-encodes PNG → JPEG

Parquet images are PNG in both datasets; the folder converter writes `image.save(f, format='JPEG')` at default quality. **The folder format is a lossy copy.** **[code]** `SimpleVLADataset` globs `*.[jp][pn][g]`, so it reads either — feed it PNGs directly if you want lossless.

### 7.7 — Instruction fields are two paraphrase sets, not a raw/clean pair

**[measured]**

| | real (1500 sample) | sim (**all 10,109**) |
|---|---|---|
| unique `instruction` | 955 / 1500 | **3,334 / 10,109** |
| unique `instruction_unified` | **257** | **874** |
| identical pairs | 74 | 2,089 |
| empty `unified` | **5** | **0** |
| words (mean) | 8.6 (4–32) | 8.1 (3–15) |

Neither is a rigid template; `instruction_unified` is simply the more canonical set. Paper describes a "Fixed Command Set" standardized per task category, expanded by GPT into an "Open Vocabulary Command Set" **[paper]** — `instruction_unified` corresponds to the former.

**Sim instructions are split by kind, and the split *is* the regime split** (§1.1) **[measured]**: **41.7% primitive** (4,218) — stating explicit metres/degrees ("Move 10.0 meters at 60 degrees to the left"), which is what makes the unit test in §1 possible — and **58.3% object-interactive** (5,891, "Face the person's direction"). The classification is unambiguous: **every instruction containing a digit also carries a unit**, so "has a number" and "is a primitive" are the same set.

Objects referenced across the full dataset: **tree 2,310, person 1,196, dog 1,010, streetlight 979, car 580, sculpture 217, umbrella 109, sunshade 103.** The real data has no comparably metric instructions.

> The earlier sample reported 57.2% primitive and `car 8`. Both were sampling artifacts — the true split is **41.7%** primitive, and `car` is the 5th most common object (580), not a rarity.

There is **no motion-type label** in either dataset. `UAV-Flow-Eval/classified_instr.json` classifies only the 273 **sim test** trajectories (Turn, Move, Shift, Rotate, Surround, Ascend/Descend, Approach, Retreat, Pass, Land). The paper cites 8 motion types **[paper]**; keyword tallies on real instructions are approximate at best (~994 move/go, ~451 pass, ~326 approach, ~91 rotate, ~91 retreat, ~53 shift, ~52 surround, ~22 turn, ~16 land, ~6 ascend/descend — categories overlap).

In sim, the rotation-bearing classes are concentrated to the point of fragility: **all 1,205 turn/rotate/face instructions are from 2025-03-27**, and **1,276 of the 1,295 trajectories that rotate more than 5°** are too (§1.1). `metric.py` scores Turn and Rotate as orientation-only classes — that signal has exactly one source date. **[measured]**

### 7.8 — Sim train/test: ids are clean, but the dates overlap ⚠️

**Now verified against all 21 shards, not 3.** The headline claim survives and is stronger; the supporting argument does not. **[measured]**

- ✅ **Zero trajectory-id overlap** between the **full 10,109** sim training trajectories and the 273 eval trajectories. Previously unverified for 18 of 21 shards; now exhaustive.
- ❌ **The dates do not separate.** The earlier "train 2025-03-14 / 2025-03-27, test 2025-03-30 … 2025-05-12" held only because the 3-shard sample missed five training dates. The truth:

| | dates |
|---|---|
| **train** (10,109) | 2025-03-14, **2025-03-27**, 2025-04-30, 2025-05-01, 2025-05-02, **2025-05-07**, 2025-05-08 |
| **test** (273) | 2025-03-30, 2025-05-06, **2025-05-07**, 2025-05-12 |

  **2025-05-07 appears in both.** So the split is *not* temporal — it is a per-trajectory split, and the "temporal separation makes id overlap unlikely in the unchecked shards" reasoning was unsound. It happens not to matter, because the exhaustive id check above now covers what that argument was standing in for.

- ⚠️ **Instruction-string overlap is much higher than reported: 204 of 239 unique test instructions (85%) also appear in train**, not 33%. Still defensible as by-design rather than leakage — the same instruction is flown from different start positions **[paper]** — but at 85% the test set is near-entirely a *novel-start-position* generalization test, not a novel-language one. Note the 273 test trajectories carry only 239 unique instructions.

## 8. Timing

**Real**: 5 Hz uniform sampling **[paper]**, confirmed **[measured]** — median dt **0.2000 s**, 95.75% of steps within 1% of 0.2 s; range 0.134–0.440 s; no gaps > 0.5 s, no non-positive dt.

| real | min | median | mean | max |
|---|---|---|---|---|
| frames / trajectory | 21 | 61 | 67.2 | 223 |
| duration (s) | 4.0 | 12.1 | 13.4 | 44.4 |

1500 real trajectories = 331 minutes of flight, collected 2025-04-02 … 2025-05-03 UTC.

**Sim has no timestamp column — the frame rate is not recoverable from the data.** The uniform 20 cm/step is *consistent with* 1 m/s at 5 Hz but is not evidence of it. **[measured]**

| sim (**all 10,109**) | min | median | mean | max |
|---|---|---|---|---|
| frames / trajectory | **4** | 25 | 30.9 | 379 |
| — 03-27 regime | **4** | 21 | 34.1 | **379** |
| — other 6 dates | 7 | 27 | 27.7 | 115 |

Sim trajectories are much shorter than real (median 25 vs 61 frames), and the 5th percentile is 6 frames — **some sim episodes are only 4 frames long**, which with the last action forced to zero and first/last frames repeated 5× (§9) makes them nearly degenerate as training samples. The degenerate short episodes and the long tail both come from **2025-03-27**: it holds every trajectory under 7 frames *and* the 379-frame maximum, while the object regime is tightly bounded at 7–115. **[measured]**

## 9. Folder format (after `prepare_data.py`)

```
<train_dir>/<trajectory_id>/
├── 000000.jpg        # zero-padded, JPEG (lossy re-encode of the source PNG)
├── 000001.jpg
├── ...
└── log.json          # {id, raw_logs, preprocessed_logs, instruction, instruction_unified, length}
```

`length` = `len(preprocessed_logs)`, and `SimpleVLADataset` pairs the *i*-th sorted image with the *i*-th log row — the correspondence is positional, so **do not add, remove, or rename images**. The loader also duplicates each episode's first and last frame 5× (`last_frame_repeat_count`) to bias the policy toward clean starts and stops. **[code]**

## 10. Reading the data

Read `log` once per trajectory, not once per frame (§3), and skip the `image` column when you only need trajectories — it is >99% of the bytes.

```python
import pyarrow.parquet as pq, json, numpy as np

# metadata only — fast, no image decoding
t = pq.ParquetFile(shard).read(columns=["id", "frame_idx", "log"])
logs = {i: json.loads(l) for i, l in zip(t.column("id").to_pylist(),
                                         t.column("log").to_pylist())}

raw = np.array(logs[tid]["raw_logs"])          # real: N x 7 | sim: N x 6
pre = np.array(logs[tid]["preprocessed_logs"]) # both: N x 6  [x,y,z,roll,yaw,pitch]

xyz_yaw   = raw[:, [0, 1, 2, 4]]               # yaw is index 4, NOT 3; valid for both widths
dyaw      = np.diff(np.deg2rad(raw[:, 4]))
dyaw      = (dyaw + np.pi) % (2 * np.pi) - np.pi   # REQUIRED in both datasets (§7.3)
total_rot = np.degrees(np.unwrap(np.deg2rad(pre[:, 4])))[-1]  # for rotation magnitude (§5)

# Sim only: which regime is this trajectory from? (§1.1) — it changes scale,
# discretisation, yaw convention and whether any rotation signal exists at all.
regime = "primitive" if tid.startswith("2025-03-27") else "object"
```

Reading all 21 sim shards this way (metadata only, no images) takes a few minutes and ~44 MB of unique `log` text — cheap enough that there is no reason to work from a shard sample. Reading them *with* images means 35.9 GB.

Images (256×256 PNG) decode from `image.bytes`:

```python
from PIL import Image; from io import BytesIO
img = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
```

The eval client resizes to **224×224** before POSTing to the inference server (`batch_run_act_all.py`), so the 256×256 native size is not what the model sees at inference time. **[code]**


