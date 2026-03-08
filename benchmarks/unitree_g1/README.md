# Unitree G1 Benchmark

## Overview

Capsule-collision variant of the Unitree G1 humanoid (29 DOF) for physics
benchmarking, with simulation settings matching those used for locomotion
training in [mjlab](https://github.com/mujocolab/mjlab).

Two scene variants:
- **scene_flat.xml** — flat ground plane
- **scene_hfield.xml** — random uniform heightfield terrain

## Assets

This benchmark uses mesh assets from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1).
Assets are not checked into this repository and must be fetched before running:

```bash
./fetch_assets.sh
```

This downloads 35 STL files (~20 MB total) to the `assets/` subdirectory.

## Derivation

- Robot XML is based on menagerie's `g1_mjx.xml` with solver settings tuned
  to match mjlab's training configuration:
  - `timestep=0.005`, `iterations=10`, `ls_iterations=20`
  - `integrator=implicitfast`, `eulerdamp=disable`
- Contact pairs replicate mjlab's `FULL_COLLISION` configuration:
  - condim=3 for feet with friction=0.6
  - condim=1 for self-collision
  - condim=3 for body-floor/terrain contacts
- Heightfield terrain matches mjlab's `HfRandomUniformTerrainCfg` parameters.
