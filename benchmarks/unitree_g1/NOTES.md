# Unitree G1 Ctrl Replay Benchmark — Investigation Notes

## Goal

Add a Unitree G1 walking benchmark to `mujoco_warp/benchmarks/` with a ctrl
sequence extracted from mjlab's trained tracking policy.

## Current State (WIP)

The ctrl sequence extraction is working, but **open-loop replay diverges** due
to a critical model mismatch between mjlab and the benchmark scene. This
document records all findings.

---

## What Works

- **`testspeed.py --replay_npz`**: Added NPZ loading support. The
  `_make_trajectory_from_npz` function loads ctrl/qpos/qvel/times arrays,
  sets initial state, and expands ctrl to physics timesteps via zero-order hold.

- **`fetch_assets.sh`**: Updated to fetch `unitree_g1` assets from menagerie
  and copy benchmark overlay files (scene XMLs, hfield, config).

- **Benchmark scene XMLs**: `scene_flat.xml` and `scene_hfield.xml` created
  using menagerie's `unitree_g1/g1.xml` with position actuators and a tracking
  camera.

- **Performance numbers** (with ctrl replay, 8192 worlds, all converged):
  - `unitree_g1_flat`:   ~834K steps/sec
  - `unitree_g1_hfield`: ~629K steps/sec

---

## The Model Mismatch Problem

### Root Cause

mjlab builds its MuJoCo model programmatically via `MjSpec`, resulting in a
model that differs from the menagerie XML in two critical ways:

1. **Actuator ordering is completely different.**

   | Index | mjlab                              | benchmark XML          |
   |-------|------------------------------------|------------------------|
   | 0     | robot/left_shoulder_pitch_joint    | left_hip_pitch_joint   |
   | 1     | robot/left_shoulder_roll_joint     | left_hip_roll_joint    |
   | ...   | (arms first, then legs)            | (legs first, then arms)|
   | 10    | robot/left_hip_pitch_joint         | right_ankle_pitch_joint|

   All 29 actuator indices are mismatched. Ctrl values extracted from mjlab are
   position targets for mjlab's actuator `[i]`, but when replayed on the
   benchmark scene, they're applied to a completely different joint.

2. **Actuator gains and parameters differ.**

   | Parameter | mjlab            | benchmark XML    |
   |-----------|------------------|------------------|
   | gainprm   | [14.25, 0, 0]    | [75, 0, 0]       |
   | biasprm   | [0, -14.25, -0.9]| [0, -75, -2]     |
   | ctrlrange | [-4.84, 4.42]    | [-2.53, 2.88]    |

   Even if actuator ordering were fixed, the different PD gains and ctrl
   ranges mean the same position target produces different torques.

### Consequence

The ctrl sequence from mjlab is fundamentally incompatible with the benchmark
scene XML. Even on mujoco_warp with the correct initial qpos/qvel, the robot
collapses immediately (z drops from 0.77 to 0.18 within 50 RL steps).

### What Was Verified

- **Initial state is correct**: qpos/qvel from the NPZ match the mjlab state
  at the capture point (t=40s). The `_make_trajectory_from_npz` function
  correctly sets mjd.qpos/qvel before `put_data` creates the warp arrays.

- **ctrl_noise kernel**: The benchmark's `ctrl_noise` kernel applies
  Ornstein-Uhlenbeck smoothing (`rate ≈ 0.95`) even with `ctrlnoisestd=0`.
  Tested with both ctrl_noise and direct ctrl assignment — both diverge.

- **Encoder bias**: Confirmed zero when domain randomization is disabled.

- **The ctrl pipeline in mjlab** is:
  1. Policy outputs raw action (29-dim)
  2. `process_actions`: `processed = raw * scale + offset`
  3. `apply_actions`: `target = processed - encoder_bias`
  4. `set_joint_position_target(target)` → writes to entity data
  5. `BuiltinPositionActuator.compute()` passes position_target through
  6. Entity writes to `sim.data.ctrl` (via actuator group ctrl_ids mapping)
  7. `sim.step()` runs decimation=4 physics steps with that ctrl

---

## Next Steps to Fix

Two viable approaches:

### Option A: Use mjlab's model for the benchmark scene (recommended)
- Export mjlab's compiled MjModel (via MjSpec.to_xml() or binary save)
- Use that XML as the benchmark scene instead of menagerie's
- Ctrl sequence would be directly compatible
- Need to figure out how to export mjlab's spec (no `_spec` attr found yet)

### Option B: Remap ctrl to match benchmark actuator ordering
- Build a name-based mapping from mjlab actuator names → benchmark indices
- Remap ctrl array columns accordingly
- **Still won't work** because actuator gains differ — a position target of
  1.0 rad produces very different torques with kp=14.25 vs kp=75

### Option C: Make the benchmark scene match mjlab's actuator config
- Modify the benchmark scene XML to use mjlab's actuator gains and ordering
- This is essentially Option A without needing the full mjlab export

---

## Files Modified

### `mujoco_warp/testspeed.py`
- Added `--replay_npz` flag
- Added `_make_trajectory_from_npz()` function
- NPZ replay takes precedence over XML keyframe replay in `_main()`

### `benchmarks/unitree_g1/` (new)
- `scene_flat.xml` — flat ground scene from menagerie g1
- `scene_hfield.xml` — heightfield terrain variant
- `shuffle_dance.npz` — captured ctrl from mjlab (t=40-45s, 250 RL steps)
- `scene.hfield` — terrain heightfield data

### `benchmarks/fetch_assets.sh` (new)
- Script to fetch menagerie assets and assemble benchmark directory

### `benchmarks/config.txt`, `benchmarks/nightly.sh`, `benchmarks/backfill.sh`
- Added unitree_g1 benchmark entries

---

## Extraction Script

Located at `/tmp/extract_ctrl.py`. Key parameters:
- Warmup: 2000 RL steps (40 seconds) before capture
- Capture: 250 RL steps (5 seconds, 1000 physics steps)
- Domain randomization disabled (`env_cfg.events = {}`)
- Terminations disabled (`env_cfg.terminations = {}`)
