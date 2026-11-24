import os
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/masterpilotguy/.cache/uv/archive-v0/tLZhQj3qOI22rSg2")
sys.path.insert(0, "/home/masterpilotguy/Open_Duck_Playground_AMD")

import numpy as np
import trimesh
import fast_simplification
import mujoco
import jax
import brax

repo_root = "/home/masterpilotguy/Open_Duck_Playground_AMD"
src_dir = os.path.join(repo_root, "playground/my_duck_mini/xmls/my_bdx_model/assets_highres_backup")
dst_dirs = [
    os.path.join(repo_root, "playground/my_duck_mini/xmls/assets"),
    os.path.join(repo_root, "playground/my_duck_mini/xmls/my_bdx_model/assets"),
]

xml_path = os.path.join(repo_root, "playground/my_duck_mini/xmls/open_duck_mini_v2.xml")
tree = ET.parse(xml_path)
root = tree.getroot()
mesh_name_to_file = {}
for m in root.findall(".//asset/mesh"):
    f = m.get("file")
    if f:
        name = m.get("name") or os.path.splitext(os.path.basename(f))[0]
        mesh_name_to_file[name] = os.path.basename(f)

file_inst_count = {}
for g in root.findall(".//worldbody//geom"):
    mname = g.get("mesh")
    if mname and mname in mesh_name_to_file:
        fname = mesh_name_to_file[mname]
        file_inst_count[fname] = file_inst_count.get(fname, 0) + 1


def weld_mesh(mesh, eps_m):
    if eps_m <= 0:
        return mesh
    v_q = np.round(mesh.vertices / eps_m) * eps_m
    return trimesh.Trimesh(vertices=v_q, faces=mesh.faces, process=True)


def try_qem(mesh, step_red, agg, normalize=True):
    v = mesh.vertices.copy()
    if normalize:
        center = v.mean(axis=0)
        scale = np.abs(v - center).max()
        if scale <= 0:
            return None
        v_in = ((v - center) / scale).astype(np.float32)
        v_dec, f_dec = fast_simplification.simplify(
            v_in, mesh.faces.astype(np.int32), target_reduction=float(step_red), agg=agg
        )
        v_out = v_dec * scale + center
    else:
        v_dec, f_dec = fast_simplification.simplify(
            v.astype(np.float32), mesh.faces.astype(np.int32), target_reduction=float(step_red), agg=agg
        )
        v_out = v_dec
    return trimesh.Trimesh(vertices=v_out, faces=f_dec, process=True)


stl_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".stl"))
rows = []
total_orig_unique = 0
total_new_unique = 0
total_orig_inst = 0
total_new_inst = 0

for fname in stl_files:
    inst = file_inst_count.get(fname, 1)
    if fname == "foot_bottom_tpu.stl":
        foot_p = os.path.join(dst_dirs[0], fname)
        foot_m = trimesh.load(foot_p, process=False)
        nf = len(foot_m.faces)
        total_orig_unique += nf
        total_new_unique += nf
        total_orig_inst += nf * inst
        total_new_inst += nf * inst
        rows.append((fname, inst, nf, nf, 1.0, 1.0, 0.0, "preserved_foot_collision"))
        continue

    src_path = os.path.join(src_dir, fname)
    orig_mesh = trimesh.load(src_path, process=True)
    orig_faces = len(orig_mesh.faces)
    orig_area = float(orig_mesh.area)
    orig_hull_area = float(orig_mesh.convex_hull.area) if orig_faces >= 4 else orig_area
    orig_bounds = orig_mesh.bounds.copy()

    if "ruthex" in fname:
        chosen = orig_mesh.convex_hull
        note = "convex_hull_insert"
    elif orig_faces <= 2500:
        chosen = orig_mesh
        note = "kept_small"
    else:
        if orig_faces <= 15000:
            target_faces = max(800, int(orig_faces * 0.25))
        elif orig_faces <= 40000:
            target_faces = max(1800, int(orig_faces * 0.18))
        else:
            target_faces = max(3500, int(orig_faces * 0.14))

        cur_mesh = orig_mesh
        note = "kept_orig"

        def is_valid(cand):
            if cand is None or len(cand.faces) < 50:
                return False
            a_rat = float(cand.area) / orig_area if orig_area > 0 else 1.0
            b_diff = float(np.abs(cand.bounds - orig_bounds).max())
            if a_rat < 0.78 or b_diff > 0.009:
                return False
            h_rat = float(cand.convex_hull.area) / orig_hull_area if orig_hull_area > 0 else 1.0
            return h_rat >= 0.93

        # Progressive sub-mm welding + multi-pass QEM ladder
        for eps_mm in [0.0, 0.05, 0.15, 0.30, 0.50, 0.80, 1.20]:
            if len(cur_mesh.faces) <= int(target_faces * 1.20):
                break
            if eps_mm > 0:
                w_cand = weld_mesh(cur_mesh, eps_mm * 1e-3)
                if is_valid(w_cand) and len(w_cand.faces) < len(cur_mesh.faces):
                    cur_mesh = w_cand
                    note = f"weld({eps_mm}mm)"
            for agg in [7, 8]:
                if len(cur_mesh.faces) <= int(target_faces * 1.20):
                    break
                desired_ratio = target_faces / float(len(cur_mesh.faces))
                step_red = float(np.clip(1.0 - desired_ratio, 0.35, 0.78))
                best_step = None
                for norm_flag in [True, False]:
                    for r_try in [step_red, 0.50]:
                        try:
                            c = try_qem(cur_mesh, r_try, agg, normalize=norm_flag)
                            if is_valid(c) and len(c.faces) < len(cur_mesh.faces):
                                if best_step is None or len(c.faces) < len(best_step.faces):
                                    best_step = c
                        except Exception:
                            pass
                if best_step is not None:
                    cur_mesh = best_step
                    note = f"qem(eps={eps_mm}mm,agg={agg})"

        chosen = cur_mesh

    new_faces = len(chosen.faces)
    area_ratio = float(chosen.area) / orig_area if orig_area > 0 else 1.0
    hull_ratio = float(chosen.convex_hull.area) / orig_hull_area if orig_hull_area > 0 else 1.0
    bounds_diff_mm = float(np.abs(chosen.bounds - orig_bounds).max()) * 1000.0

    for d in dst_dirs:
        chosen.export(os.path.join(d, fname))

    total_orig_unique += orig_faces
    total_new_unique += new_faces
    total_orig_inst += orig_faces * inst
    total_new_inst += new_faces * inst
    rows.append((fname, inst, orig_faces, new_faces, area_ratio, hull_ratio, bounds_diff_mm, note))

print(f"{'Filename':<45} {'Inst':>4} {'Orig':>7} {'New':>7} {'AreaRat':>8} {'HullRat':>8} {'Bnd(mm)':>8} {'Method'}")
print("-" * 112)
for r in rows:
    print(f"{r[0]:<45} {r[1]:>4} {r[2]:>7} {r[3]:>7} {r[4]:>8.4f} {r[5]:>8.4f} {r[6]:>8.3f} {r[7]}")
print("-" * 112)
print(f"Total Unique Faces   : {total_orig_unique:,} -> {total_new_unique:,} ({total_new_unique/total_orig_unique*100:.1f}%)")
print(f"Total Instanced Faces: {total_orig_inst:,} -> {total_new_inst:,} ({total_new_inst/total_orig_inst*100:.1f}%)")
total_bytes = sum(os.path.getsize(os.path.join(dst_dirs[0], f)) for f in os.listdir(dst_dirs[0]) if f.endswith(".stl"))
print(f"Total STL Disk Size  : {total_bytes / (1024*1024):.2f} MB")

for d in dst_dirs:
    fp = os.path.join(d, "foot_bottom_tpu.stl")
    fm = trimesh.load(fp, process=False)
    print(f"Verified {fp}: faces={len(fm.faces)}, bytes={os.path.getsize(fp)}")
    assert len(fm.faces) == 32 and os.path.getsize(fp) == 1684

scenes = [
    "playground/my_duck_mini/xmls/scene_flat_terrain.xml",
    "playground/my_duck_mini/xmls/scene_rough_terrain.xml",
    "playground/my_duck_mini/xmls/scene_flat_terrain_backlash.xml",
    "playground/my_duck_mini/xmls/my_bdx_model/scene_flat_terrain.xml",
]
for s in scenes:
    sp = os.path.join(repo_root, s)
    t0 = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(sp)
    dt_load = (time.perf_counter() - t0) * 1000.0
    print(f"Loaded {s} in {dt_load:.1f} ms: nmesh={m.nmesh}, nmeshvert={m.nmeshvert:,}, nmeshface={m.nmeshface:,}")

# Benchmark MjInfer 500 steps headless using exact MjInfer.run() step logic
from playground.my_duck_mini.mujoco_infer import MjInfer
onnx_candidates = []
for root_d, _, files in os.walk(repo_root):
    for f in files:
        if f.endswith(".onnx"):
            onnx_candidates.append(os.path.join(root_d, f))
onnx_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
if onnx_candidates:
    ref_data = os.path.join(repo_root, "playground/my_duck_mini/data/polynomial_coefficients.pkl")
    model_xml = os.path.join(repo_root, "playground/my_duck_mini/xmls/scene_flat_terrain.xml")
    infer = MjInfer(model_xml, ref_data, onnx_candidates[0], standing=False)
    t0 = time.perf_counter()
    for _ in range(500):
        infer.imitation_i += 1.0 * infer.phase_frequency_factor
        infer.imitation_i = infer.imitation_i % infer.PRM.nb_steps_in_period
        infer.imitation_phase = np.array([
            np.cos(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
            np.sin(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
        ])
        obs = infer.get_obs(infer.data, infer.commands)
        action = infer.policy.infer(obs)
        infer.last_last_last_action = infer.last_last_action.copy()
        infer.last_last_action = infer.last_action.copy()
        infer.last_action = action.copy()
        infer.motor_targets = infer.default_actuator + action * infer.action_scale
        infer.data.ctrl = infer.motor_targets.copy()
        for _ in range(infer.decimation):
            mujoco.mj_step(infer.model, infer.data)
    dt = time.perf_counter() - t0
    print(f"MjInfer 500-step headless benchmark ({os.path.basename(onnx_candidates[0])}): {dt*1000:.2f} ms total ({dt/500*1000:.3f} ms/step, {500/dt:.1f} Hz)")

print(f"Env Check: jax={jax.__version__}, devices={jax.devices()}, mujoco={mujoco.__version__}, brax={brax.__version__}")

