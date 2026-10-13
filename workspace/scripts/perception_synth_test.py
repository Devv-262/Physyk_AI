"""Offline accuracy test for physyk_perception/color_segment.py — no simulator needed.

Ray-casts the overview camera (same intrinsics/extrinsics as isaac_sim_service.py) against boxes
for the cubes and tray to synthesize an RGB frame and an exact depth frame, then compares what the
color detector recovers against the known positions, for random scattered layouts and for
stacks. Run:   .venv/bin/python workspace/scripts/perception_synth_test.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from physyk_perception import color_segment as cs  # noqa: E402

FX = 22.0 / 20.955 * 960
K = {"fx": FX, "fy": FX, "cx": 480.0, "cy": 270.0, "width": 960, "height": 540}
EYE = np.array([1.85, -1.55, 1.30])
TGT = np.array([0.35, 0.0, 0.28])
_f = TGT - EYE
_f /= np.linalg.norm(_f)
_r = np.cross(_f, [0, 0, 1.0])
_r /= np.linalg.norm(_r)
_u = np.cross(_r, _f)
T = np.eye(4)
T[:3, :3] = np.column_stack([_r, _u, -_f])
T[:3, 3] = EYE
H, W = 540, 960
VV, UU = np.mgrid[0:H, 0:W]
RAYS = np.stack([(UU - 480.0) / FX, -(VV - 270.0) / FX, -np.ones((H, W))], -1) @ T[:3, :3].T
# Per-face colors MEASURED from a live overview frame (studio lighting), as OpenCV HSV.
# The top face renders nearly white (low saturation) — that is what broke a hue+saturation mask.
_HSV = {  # top, +x side, -y side
    "red_cube":   [(1, 56, 242), (1, 93, 226), (2, 121, 201)],
    "green_cube": [(78, 47, 246), (78, 97, 224), (76, 129, 201)],
    "blue_cube":  [(98, 40, 247), (100, 86, 228), (101, 123, 206)],
}


def _rgb(hsv):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0])


FACE_RGB = {k: [_rgb(h) for h in v] for k, v in _HSV.items()}
COL = {"red_cube": None, "green_cube": None, "blue_cube": None, "target_tray": (239, 235, 146)}
CUBE_HALF = [0.025] * 3
TRAY = ((0.42, -0.32, 0.015), [0.11, 0.12, 0.015])
rng = np.random.default_rng(1)


def _hit(center, half):
    """Ray-box hit. Returns (z-depth per pixel, entering-face index 0=top 1=+x side 2=-y side/other)."""
    lo, hi = np.array(center) - half, np.array(center) + half
    t1, t2 = (lo - EYE) / RAYS, (hi - EYE) / RAYS
    per_axis = np.minimum(t1, t2)
    tmin, tmax = per_axis.max(-1), np.maximum(t1, t2).min(-1)
    axis = per_axis.argmax(-1)
    face = np.where(axis == 2, 0, np.where(axis == 0, 1, 2))
    return np.where(tmax >= np.maximum(tmin, 0), tmin, np.inf), face


def render(objs):
    rgb = np.full((H, W, 3), (203, 208, 214), np.uint8)
    depth = np.full((H, W), np.inf)
    for k, (c, half) in objs.items():
        t, face = _hit(c, np.array(half))
        m = t < depth
        depth = np.where(m, t, depth)
        if COL[k] is not None:
            rgb[m] = COL[k]
        else:
            for fi in range(3):
                rgb[m & (face == fi)] = FACE_RGB[k][fi]
    depth[~np.isfinite(depth)] = 0
    return rgb, depth.astype(np.float32)


def scatter():
    o, pts = {"target_tray": TRAY}, []
    for k in ("red_cube", "green_cube", "blue_cube"):
        while True:
            p = (rng.uniform(0.28, 0.65), rng.uniform(-0.15, 0.55))
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.10 for q in pts):
                break
        pts.append(p)
        o[k] = ((p[0], p[1], 0.025), CUBE_HALF)
    return o


def stack():
    o, base = {"target_tray": TRAY}, (0.42, -0.32)
    for i, k in enumerate(("red_cube", "blue_cube", "green_cube")):
        o[k] = ((base[0] + rng.normal(0, 0.004), base[1] + rng.normal(0, 0.004), 0.03 + 0.025 + 0.05 * i), CUBE_HALF)
    return o


def run(layouts, label):
    roi = cs.roi_mask((H, W), K, T)
    errs, missed = {k: [] for k in COL}, 0
    for objs in layouts:
        rgb, depth = render(objs)
        for k in COL:
            r = cs.locate_cube(rgb, depth, k, K, T, roi=roi)
            if r is None:
                missed += 1
                continue
            errs[k].append(np.linalg.norm(np.array(r["position"]) - np.array(objs[k][0])) * 1000)
    print(f"{label}: missed={missed}")
    for k, v in errs.items():
        if v:
            print(f"   {k:12s} n={len(v):2d} mean={np.mean(v):5.1f}mm p90={np.percentile(v, 90):5.1f}mm max={np.max(v):5.1f}mm")


if __name__ == "__main__":
    run([scatter() for _ in range(30)], "SCATTERED (30 layouts)")
    run([stack() for _ in range(20)], "STACKED in tray (20 layouts)")
