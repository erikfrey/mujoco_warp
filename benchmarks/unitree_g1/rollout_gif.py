#!/usr/bin/env python3
"""Visualize control noise by overlapping N transparent robot rollouts.

Simulates N parallel worlds using mujoco_warp with control noise, then
renders all robots overlapping in a single MuJoCo scene. The reference
robot (world 0) is rendered fully opaque with zero noise; remaining robots
use semi-transparent "ghost" materials.

Requires: pip install Pillow  (not a mujoco_warp dependency)

Usage:
  python rollout_gif.py CLIP.npz [options]

Examples:
  # 32 robots, noise=0.01, alpha=0.15, flat terrain
  python rollout_gif.py clip.npz

  # 64 robots on heightfield terrain with higher noise
  python rollout_gif.py clip.npz --nworld 64 --noise 0.02 --terrain hfield

  # Custom alpha and output path
  python rollout_gif.py clip.npz --alpha 0.1 --output my_rollout.gif
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from PIL import Image

from mujoco_warp._src.forward import step as mjw_step
from mujoco_warp._src.io import put_model, put_data

wp.init()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: assets deployed to /tmp by testspeed.py; fall back to script dir
DEFAULT_BENCHMARK_DIR = "/tmp/mujoco_warp_benchmarks/unitree_g1"
if not os.path.isdir(DEFAULT_BENCHMARK_DIR):
    DEFAULT_BENCHMARK_DIR = SCRIPT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="Path to NPZ clip file (must contain ctrl, qpos, qvel)")
    parser.add_argument("--nworld", type=int, default=32, help="Number of parallel worlds (default: 32)")
    parser.add_argument("--noise", type=float, default=0.01, help="Control noise scale (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.15, help="Ghost robot alpha (default: 0.15)")
    parser.add_argument("--terrain", choices=["flat", "hfield"], default="flat",
                        help="Terrain type (default: flat)")
    parser.add_argument("--output", "-o", default=None, help="Output GIF path (default: overlap_<N>w.gif)")
    parser.add_argument("--width", type=int, default=640, help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Render height (default: 480)")
    parser.add_argument("--render-every", type=int, default=10,
                        help="Render every N physics steps (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR,
                        help="Path to unitree_g1 benchmark dir (default: auto-detected)")
    return parser.parse_args()


def rename_tree(elem, suffix):
    """Recursively rename all named attributes in an XML element tree."""
    for attr in ["name", "joint", "site", "body1", "body2", "geom1", "geom2"]:
        val = elem.get(attr)
        if val is not None:
            elem.set(attr, f"{val}_{suffix}")
    for child in elem:
        rename_tree(child, suffix)


def get_robot_index(mjm, geom_id):
    """Return robot index (0..N-1) for a geom, or -1 if not a robot geom."""
    body_id = mjm.geom_bodyid[geom_id]
    while body_id > 0:
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("pelvis_r"):
            return int(name.split("_r")[1])
        body_id = mjm.body_parentid[body_id]
    return -1


def build_overlap_scene(robot_xml_path, nworld, alpha, terrain, hfield_path=None):
    """Build a MuJoCo XML with N overlapping robots and ghost materials."""
    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()
    pelvis = robot_root.find(".//body[@name='pelvis']")
    orig_pos = pelvis.get("pos")
    defaults_elem = robot_root.find("default")
    asset_elem = robot_root.find("asset")
    meshdir = os.path.join(os.path.dirname(robot_xml_path), "assets")

    combined = '<mujoco model="overlap">\n'
    combined += f'  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>\n'

    option_elem = robot_root.find("option")
    if option_elem is not None:
        combined += "  " + ET.tostring(option_elem, encoding="unicode") + "\n"

    combined += """  <visual>
    <map znear="0.01" zfar="200"/>
    <quality shadowsize="8192"/>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
"""

    # Copy defaults exactly from original
    combined += "  " + ET.tostring(defaults_elem, encoding="unicode") + "\n"

    # Asset section
    combined += "  <asset>\n"
    combined += '    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>\n'
    combined += '    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>\n'
    combined += '    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>\n'
    combined += f'    <material name="silver_ghost" rgba="0.7 0.7 0.7 {alpha}"/>\n'
    combined += f'    <material name="black_ghost" rgba="0.2 0.2 0.2 {alpha}"/>\n'
    for child in asset_elem:
        combined += "    " + ET.tostring(child, encoding="unicode") + "\n"

    if terrain == "hfield":
        combined += f'    <hfield name="terrain" size="4.0 4.0 0.1 0.05" file="{hfield_path}"/>\n'
        combined += """  </asset>
  <worldbody>
    <geom name="terrain_hfield" type="hfield" hfield="terrain"
          pos="0.0 0.0 -0.05" contype="0" conaffinity="0"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
"""
    else:
        combined += """  </asset>
  <worldbody>
    <geom name="floor" size="0 0 .05" type="plane" material="grid"
          contype="0" conaffinity="0"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
"""

    for idx in range(nworld):
        robot = copy.deepcopy(pelvis)
        robot.set("pos", orig_pos)
        rename_tree(robot, f"r{idx}")
        combined += "    " + ET.tostring(robot, encoding="unicode") + "\n"

    combined += "  </worldbody>\n</mujoco>"
    return combined


def configure_rendering_model(grid_mjm, nworld):
    """Disable contacts, extra lights, and remap ghost materials."""
    # Disable all contacts (rendering-only model)
    for i in range(grid_mjm.ngeom):
        grid_mjm.geom_contype[i] = 0
        grid_mjm.geom_conaffinity[i] = 0

    # Disable duplicate trackcom lights (keep only r0's)
    disabled_lights = 0
    for i in range(grid_mjm.nlight):
        body_id = grid_mjm.light_bodyid[i]
        name = mujoco.mj_id2name(grid_mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("pelvis_r") and not name.endswith("_r0"):
            grid_mjm.light_active[i] = 0
            disabled_lights += 1
    print(f"  Disabled {disabled_lights} duplicate lights")

    # Remap noisy robots (r1+) to ghost materials; r0 keeps full opacity
    silver_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "silver")
    black_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "black")
    sg_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "silver_ghost")
    bg_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "black_ghost")

    remapped = 0
    for i in range(grid_mjm.ngeom):
        if grid_mjm.geom_group[i] != 2:  # only visual geoms
            continue
        ridx = get_robot_index(grid_mjm, i)
        if ridx <= 0:  # skip r0 and non-robot geoms
            continue
        if grid_mjm.geom_matid[i] == silver_id:
            grid_mjm.geom_matid[i] = sg_id
            remapped += 1
        elif grid_mjm.geom_matid[i] == black_id:
            grid_mjm.geom_matid[i] = bg_id
            remapped += 1
    print(f"  Remapped {remapped} geoms to ghost materials")


@wp.kernel
def _set_ctrl_noisy(
    target: wp.array1d(dtype=float),
    noise: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    wid, aid = wp.tid()
    out[wid, aid] = target[aid] + noise[wid, aid]


def main():
    args = parse_args()
    nworld = args.nworld
    terrain = args.terrain

    # Load clip
    npz = np.load(args.clip)
    ctrl_seq = npz["ctrl"]
    print(f"Clip: {len(ctrl_seq)} steps, nu={ctrl_seq.shape[1]}")

    # Physics model
    bdir = args.benchmark_dir
    scene_xml = os.path.join(bdir,
                             "scene_hfield.xml" if terrain == "hfield" else "scene_flat.xml")
    robot_xml = os.path.join(bdir, "unitree_g1_mjlab.xml")
    hfield_path = os.path.join(bdir, "hfield.png")

    mjm = mujoco.MjModel.from_xml_path(scene_xml)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = npz["qpos"][0]
    mjd.qvel[:] = npz["qvel"][0]
    mujoco.mj_forward(mjm, mjd)
    m = put_model(mjm)
    d = put_data(mjm, mjd, nworld=nworld, nconmax=48, njmax=192)

    # Build rendering model
    print(f"Building {nworld}-robot overlap model (alpha={args.alpha})...")
    scene_xml_str = build_overlap_scene(robot_xml, nworld, args.alpha,
                                        terrain, hfield_path)
    tmp_xml = "/tmp/overlap_scene.xml"
    with open(tmp_xml, "w") as f:
        f.write(scene_xml_str)

    grid_mjm = mujoco.MjModel.from_xml_path(tmp_xml)
    grid_mjd = mujoco.MjData(grid_mjm)
    print(f"  nq={grid_mjm.nq}, ngeom={grid_mjm.ngeom}, nmat={grid_mjm.nmat}")
    configure_rendering_model(grid_mjm, nworld)

    # Map joint addresses
    world_nq = mjm.nq
    world_qposadr = []
    for idx in range(nworld):
        jid = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"floating_base_joint_r{idx}")
        world_qposadr.append(grid_mjm.jnt_qposadr[jid])

    renderer = mujoco.Renderer(grid_mjm, width=args.width, height=args.height)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = npz["qpos"][0][:3]
    cam.lookat[2] = 0.7
    cam.distance = 3.0
    cam.azimuth = 140
    cam.elevation = -20

    # Generate noise (world 0 = no noise)
    rng = np.random.default_rng(args.seed)
    all_noise = rng.normal(0, args.noise,
                           (len(ctrl_seq), nworld, mjm.nu)).astype(np.float32)
    all_noise[:, 0, :] = 0

    # Simulate and render
    print("Simulating...")
    frames = []
    step_count = 0
    for rl_step in range(len(ctrl_seq)):
        target = wp.array(ctrl_seq[rl_step], dtype=wp.float32)
        noise = wp.array(all_noise[rl_step], dtype=wp.float32)
        for sub in range(4):
            wp.launch(_set_ctrl_noisy, dim=(nworld, mjm.nu),
                      inputs=[target, noise, d.ctrl])
            mjw_step(m, d)
            step_count += 1
            if step_count % args.render_every == 0:
                wp.synchronize()
                all_qpos = d.qpos.numpy()
                for idx in range(nworld):
                    qa = world_qposadr[idx]
                    grid_mjd.qpos[qa : qa + world_nq] = all_qpos[idx]
                mujoco.mj_kinematics(grid_mjm, grid_mjd)
                mujoco.mj_camlight(grid_mjm, grid_mjd)
                renderer.update_scene(grid_mjd, camera=cam)
                frames.append(renderer.render().copy())

    # Save
    output = args.output or f"overlap_{nworld}w_{terrain}.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output, save_all=True, append_images=pil_frames[1:],
        duration=50, loop=0, optimize=True,
    )
    print(f"Saved: {output} ({os.path.getsize(output)/1024:.0f}KB), {len(frames)} frames")
    renderer.close()


if __name__ == "__main__":
    main()
