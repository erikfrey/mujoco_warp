#!/bin/bash

# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Build a runnable benchmarks directory by combining mujoco_menagerie assets
# with benchmark-specific files from this repo.
#
# The output is a self-contained directory that can be used by run.sh, nightly.sh,
# and backfill.sh to run benchmarks.
#
# Usage:
#   ./fetch_assets.sh --output-dir=<path> [--cache-dir=<path>]
#
# Required:
#   --output-dir  Where to build the assembled benchmarks directory
#
# Optional:
#   --cache-dir   Where to clone mujoco_menagerie (default: /tmp/mujoco_menagerie)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARKS_DIR="$REPO_DIR/benchmarks"
OUTPUT_DIR=""
CACHE_DIR="/tmp/mujoco_menagerie"
MENAGERIE_REPO="https://github.com/google-deepmind/mujoco_menagerie.git"

for arg in "$@"; do
    case $arg in
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --cache-dir=*)
            CACHE_DIR="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 --output-dir=<path> [--cache-dir=<path>]"
            exit 1
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    echo "Usage: $0 --output-dir=<path> [--cache-dir=<path>]"
    exit 1
fi

# --- Step 1: Clone menagerie ---

if [[ -d "$CACHE_DIR/.git" ]]; then
    echo "mujoco_menagerie already cloned at $CACHE_DIR, skipping."
else
    echo "Cloning mujoco_menagerie into $CACHE_DIR..."
    git clone --depth 1 "$MENAGERIE_REPO" "$CACHE_DIR"
fi

# --- Step 2: Copy repo benchmarks to output directory ---

echo "Copying repo benchmarks to $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"
rsync -a --exclude='fetch_assets.sh' "$BENCHMARKS_DIR/" "$OUTPUT_DIR/"

# --- Step 3: Overlay menagerie assets ---

# aloha_pot: robot STLs + table OBJs + texture from menagerie
echo "Adding aloha assets from menagerie..."
mkdir -p "$OUTPUT_DIR/aloha_pot/assets"
cp "$CACHE_DIR/aloha/assets"/* "$OUTPUT_DIR/aloha_pot/assets/"
echo "  aloha_pot: $(ls "$OUTPUT_DIR/aloha_pot/assets" | wc -l) files"

# franka_emika_panda: collision meshes from menagerie
echo "Adding franka assets from menagerie..."
mkdir -p "$OUTPUT_DIR/franka_emika_panda/assets"
cp "$CACHE_DIR/franka_emika_panda/assets"/* "$OUTPUT_DIR/franka_emika_panda/assets/"
echo "  franka_emika_panda: $(ls "$OUTPUT_DIR/franka_emika_panda/assets" | wc -l) files"

# unitree_g1: robot STLs from menagerie
echo "Adding unitree_g1 assets from menagerie..."
mkdir -p "$OUTPUT_DIR/unitree_g1/assets"
cp "$CACHE_DIR/unitree_g1/assets"/* "$OUTPUT_DIR/unitree_g1/assets/"
echo "  unitree_g1: $(ls "$OUTPUT_DIR/unitree_g1/assets" | wc -l) files"

# --- Step 4: Copy any other referenced assets ---

# aloha_sdf references SDF collision textures from the repo's test_data
SDF_ASSET_SRC="$REPO_DIR/mujoco_warp/test_data/collision_sdf/asset"
SDF_ASSET_DST="$OUTPUT_DIR/../mujoco_warp/test_data/collision_sdf/asset"
if [[ -d "$SDF_ASSET_SRC" ]]; then
    echo "Copying SDF collision textures..."
    mkdir -p "$SDF_ASSET_DST"
    cp "$SDF_ASSET_SRC"/* "$SDF_ASSET_DST/"
fi

echo "Done. Assembled benchmarks directory: $OUTPUT_DIR"
