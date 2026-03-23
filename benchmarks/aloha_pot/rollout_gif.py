#!/usr/bin/env python3
"""Visualize control noise for aloha_pot by overlapping N transparent rollouts.

Simulates N parallel worlds using mujoco_warp with control noise, then renders
all in a single multi-robot scene. The reference world (0) is fully opaque with
zero noise; remaining worlds use transparent "ghost" materials.

Requires: pip install Pillow  (not a mujoco_warp dependency)

Usage:
  MUJOCO_GL=egl python rollout_gif.py lift_pot.npz [options]
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from PIL import Image

from mujoco_warp._src.forward import step as mjw_step
from mujoco_warp._src.io import put_model, put_data

wp.init()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCHMARK_DIR = "/tmp/mujoco_warp_benchmarks/aloha_pot"
if not os.path.isdir(DEFAULT_BENCHMARK_DIR):
    DEFAULT_BENCHMARK_DIR = SCRIPT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="NPZ file with ctrl/qpos/qvel")
    parser.add_argument("--nworld", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--render-every", type=int, default=10,
                        help="Render every N physics steps (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", default="overhead_cam")
    parser.add_argument("--duration", type=int, default=40, help="GIF frame duration in ms")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    return parser.parse_args()


def rename_tree(elem, suffix):
    """Recursively rename all named attributes."""
    for attr in ["name", "joint", "site", "body1", "body2", "geom1", "geom2",
                 "target", "jointinparent"]:
        val = elem.get(attr)
        if val is not None:
            elem.set(attr, f"{val}_{suffix}")
    for child in elem:
        rename_tree(child, suffix)


def get_robot_index(mjm, geom_id, root_body_names):
    """Return robot index for a geom, or -1 if not a robot geom."""
    body_id = mjm.geom_bodyid[geom_id]
    while body_id > 0:
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for root_name, idx in root_body_names.items():
            if name == root_name:
                return idx
        body_id = mjm.body_parentid[body_id]
    return -1


def build_overlap_scene(scene_xml_path, nworld, alpha):
    """Build overlap scene by duplicating the full scene N times."""
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    # Collect all top-level bodies under worldbody
    worldbody = root.find("worldbody")
    top_bodies = list(worldbody)  # light, geom (floor), and robot bodies

    # Separate static elements from dynamic robot elements
    # Only bodies with joints (or children with joints) need duplication
    # Static bodies (table) stay as single copies
    # Lights with target attributes reference robot bodies, so they must be duplicated
    static_elems = []
    robot_bodies = []
    for elem in top_bodies:
        if elem.tag == "body":
            # Check if this body tree has any joints
            has_joint = len(elem.findall(".//joint")) > 0 or len(elem.findall("joint")) > 0
            # Also check for freejoints
            has_freejoint = len(elem.findall(".//freejoint")) > 0 or elem.find("freejoint") is not None
            if has_joint or has_freejoint:
                robot_bodies.append(elem)
            else:
                static_elems.append(elem)
        elif elem.tag == "light" and elem.get("target"):
            robot_bodies.append(elem)
        else:
            static_elems.append(elem)

    # Collect asset and default sections
    asset_elem = root.find("asset")
    defaults_elem = root.find("default")
    compiler_elem = root.find("compiler")
    option_elem = root.find("option")

    # Get meshdir from compiler
    meshdir = compiler_elem.get("meshdir", ".") if compiler_elem is not None else "."
    if not os.path.isabs(meshdir):
        meshdir = os.path.join(os.path.dirname(scene_xml_path), meshdir)

    # Build materials list for ghosting
    mat_names = set()
    if asset_elem is not None:
        for mat in asset_elem.findall("material"):
            name = mat.get("name")
            if name:
                mat_names.add(name)

    # Start building XML
    combined = '<mujoco model="overlap">\n'
    if compiler_elem is not None:
        # Override meshdir and texturedir to absolute paths
        compiler_copy = copy.deepcopy(compiler_elem)
        compiler_copy.set("meshdir", meshdir)
        texturedir = compiler_elem.get("texturedir", os.path.dirname(scene_xml_path))
        if not os.path.isabs(texturedir):
            texturedir = os.path.join(os.path.dirname(scene_xml_path), texturedir)
        compiler_copy.set("texturedir", texturedir)
        combined += "  " + ET.tostring(compiler_copy, encoding="unicode") + "\n"

    if option_elem is not None:
        combined += "  " + ET.tostring(option_elem, encoding="unicode") + "\n"

    combined += """  <visual>
    <map znear="0.01" zfar="200"/>
    <quality shadowsize="8192"/>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
"""

    if defaults_elem is not None:
        combined += "  " + ET.tostring(defaults_elem, encoding="unicode") + "\n"

    # Asset section with ghost materials
    combined += "  <asset>\n"
    if asset_elem is not None:
        for child in asset_elem:
            combined += "    " + ET.tostring(child, encoding="unicode") + "\n"
    # Add ghost variants of each material
    for mat_name in sorted(mat_names):
        if mat_name == "groundplane":
            continue  # don't ghost the floor
        mid = mujoco.MjModel.from_xml_string(
            f'<mujoco><asset><material name="x" rgba="1 1 1 1"/></asset></mujoco>'
        )  # dummy — we'll get the real rgba from the compiled model later
        combined += f'    <material name="{mat_name}_ghost" rgba="0.5 0.5 0.5 {alpha}"/>\n'
    combined += "  </asset>\n"

    # Worldbody: static elements + N copies of robot bodies
    combined += "  <worldbody>\n"
    for elem in static_elems:
        combined += "    " + ET.tostring(elem, encoding="unicode") + "\n"

    for idx in range(nworld):
        for body in robot_bodies:
            body_copy = copy.deepcopy(body)
            rename_tree(body_copy, f"r{idx}")
            combined += "    " + ET.tostring(body_copy, encoding="unicode") + "\n"

    combined += "  </worldbody>\n"

    combined += "</mujoco>"
    return combined, sorted(mat_names - {"groundplane", "table"})


def configure_rendering_model(grid_mjm, nworld, ghost_mat_names, robot_body_prefix, alpha):
    """Disable contacts, extra lights, remap ghost materials."""
    for i in range(grid_mjm.ngeom):
        grid_mjm.geom_contype[i] = 0
        grid_mjm.geom_conaffinity[i] = 0

    # Disable duplicate lights (keep only r0's)
    for i in range(grid_mjm.nlight):
        name = mujoco.mj_id2name(grid_mjm, mujoco.mjtObj.mjOBJ_LIGHT, i) or ""
        # Keep lights ending in _r0, disable _r1 through _r{N-1}
        if "_r" in name and not name.endswith("_r0"):
            grid_mjm.light_active[i] = 0

    # Build root body name → index mapping
    root_bodies = {}
    for idx in range(nworld):
        for i in range(grid_mjm.nbody):
            name = mujoco.mj_id2name(grid_mjm, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for prefix in robot_body_prefix:
                if name == f"{prefix}_r{idx}":
                    root_bodies[name] = idx

    # Fix ghost material rgba to match real material rgb
    for mat_name in ghost_mat_names:
        real_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        ghost_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, f"{mat_name}_ghost")
        if real_id >= 0 and ghost_id >= 0:
            grid_mjm.mat_rgba[ghost_id, :3] = grid_mjm.mat_rgba[real_id, :3]

    # Remap non-r0 visual geoms to ghost materials
    mat_mapping = {}
    for mat_name in ghost_mat_names:
        real_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        ghost_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_MATERIAL, f"{mat_name}_ghost")
        if real_id >= 0 and ghost_id >= 0:
            mat_mapping[real_id] = ghost_id

    remapped = 0
    for i in range(grid_mjm.ngeom):
        ridx = get_robot_index(grid_mjm, i, root_bodies)
        if ridx <= 0:
            continue
        if grid_mjm.geom_matid[i] in mat_mapping:
            grid_mjm.geom_matid[i] = mat_mapping[grid_mjm.geom_matid[i]]
            remapped += 1
    print(f"  Remapped {remapped} geoms to ghost materials")

    # Hide static base geoms for r1+ (base_link has no joint, its geoms are fixed in
    # place — 64 copies cause z-fighting)
    hidden = 0
    for i in range(grid_mjm.ngeom):
        body_id = grid_mjm.geom_bodyid[i]
        bname = mujoco.mj_id2name(grid_mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        # Check if this is a base_link body for a non-r0 robot
        if "base_link" in bname and "_r" in bname and not bname.endswith("_r0"):
            # Check body has no joint (it's static)
            has_joint = any(grid_mjm.jnt_bodyid[j] == body_id for j in range(grid_mjm.njnt))
            if not has_joint:
                grid_mjm.geom_rgba[i, 3] = 0  # fully transparent
                hidden += 1
    if hidden:
        print(f"  Hidden {hidden} static base geoms")

    # Override pot geom colors to uniform silver (original has random per-geom colors)
    silver = np.array([0.75, 0.75, 0.78, 1.0], dtype=np.float32)
    silver_ghost = np.array([0.75, 0.75, 0.78, alpha], dtype=np.float32)
    for i in range(grid_mjm.ngeom):
        body_id = grid_mjm.geom_bodyid[i]
        bname = mujoco.mj_id2name(grid_mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "partnet" in bname:
            ridx = -1
            for root_name, idx in root_bodies.items():
                if "partnet" in root_name and bname.startswith(root_name.split("_r")[0]):
                    ridx = idx
                    break
            grid_mjm.geom_rgba[i] = silver_ghost if ridx > 0 else silver


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

    npz = np.load(args.clip)
    ctrl_seq = npz["ctrl"]
    print(f"Clip: {len(ctrl_seq)} steps, nu={ctrl_seq.shape[1]}")

    scene_xml = os.path.join(args.benchmark_dir, "scene.xml")

    # Physics: use original scene with mujoco_warp
    mjm = mujoco.MjModel.from_xml_path(scene_xml)
    mjd = mujoco.MjData(mjm)
    if "qpos" in npz:
        mjd.qpos[:] = npz["qpos"][0] if npz["qpos"].ndim > 1 else npz["qpos"]
    if "qvel" in npz:
        mjd.qvel[:] = npz["qvel"][0] if npz["qvel"].ndim > 1 else npz["qvel"]
    mujoco.mj_forward(mjm, mjd)
    m = put_model(mjm)
    d = put_data(mjm, mjd, nworld=nworld, nconmax=24, njmax=128)

    # Build rendering model
    print(f"Building {nworld}-robot overlap model (alpha={args.alpha})...")
    scene_xml_str, ghost_mat_names = build_overlap_scene(scene_xml, nworld, args.alpha)
    tmp_xml = "/tmp/overlap_aloha_scene.xml"
    with open(tmp_xml, "w") as f:
        f.write(scene_xml_str)

    grid_mjm = mujoco.MjModel.from_xml_path(tmp_xml)
    grid_mjd = mujoco.MjData(grid_mjm)
    print(f"  nq={grid_mjm.nq}, ngeom={grid_mjm.ngeom}, nmat={grid_mjm.nmat}")

    # Find robot body prefixes (top-level bodies under worldbody that aren't static)
    robot_prefixes = []
    for i in range(mjm.nbody):
        if mjm.body_parentid[i] == 0 and i > 0:
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if name:
                robot_prefixes.append(name)

    configure_rendering_model(grid_mjm, nworld, ghost_mat_names, robot_prefixes, args.alpha)

    # Map qpos for each world to the rendering model
    world_nq = mjm.nq
    world_qposadr = []
    # The rendering model has N copies of the full qpos
    for idx in range(nworld):
        world_qposadr.append(idx * world_nq)

    renderer = mujoco.Renderer(grid_mjm, width=args.width, height=args.height)

    # Use a named camera
    cam_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_CAMERA, f"{args.camera}_r0")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(grid_mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    cam = mujoco.MjvCamera()
    if cam_id >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.5
        cam.azimuth = 180
        cam.elevation = -30

    # Noise
    rng = np.random.default_rng(args.seed)
    all_noise = rng.normal(0, args.noise,
                           (len(ctrl_seq), nworld, mjm.nu)).astype(np.float32)
    all_noise[:, 0, :] = 0

    # Simulate and render
    print("Simulating...")
    frames = []
    step_count = 0
    for step_idx in range(len(ctrl_seq)):
        target = wp.array(ctrl_seq[step_idx], dtype=wp.float32)
        noise = wp.array(all_noise[step_idx], dtype=wp.float32)
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

    output = args.output or f"overlap_{nworld}w_aloha.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output, save_all=True, append_images=pil_frames[1:],
        duration=args.duration, loop=0, optimize=True,
    )
    print(f"Saved: {output} ({os.path.getsize(output)/1024:.0f}KB), {len(frames)} frames")
    renderer.close()


if __name__ == "__main__":
    main()
