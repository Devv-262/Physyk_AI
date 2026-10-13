"""Vessel detector for the shelf, from the rack camera (RGB + depth).

The overview camera models the table plane, so a vessel standing on a shelf tier is measured at the wrong height (or not
found at all) and gets no bounding box. The rack camera sees the shelf from the side, so here the reference surface is
each tier's own plank: every depth pixel is deprojected to world coordinates, and the points between 1.5 cm and 13 cm
above a plank (and inside that tier's footprint) are vessel candidates. Candidates are grouped into connected
components in the image; each component gives

  * the vessel kind from its height above the plank (flask 9.5 cm, beaker 7.0 cm),
  * its liquid colour (same colour naming as glass_detect), hence its registry key,
  * its base-centre position in world coordinates (visible-surface centroid, shifted away from the camera by half
    a vessel radius, because the camera only sees the near side),
  * its pixel bounding box.

Pure numpy/cv2, no simulator imports, so it is testable on saved frames.
"""

import cv2
import numpy as np

from .glass_detect import VESSEL_KEYS, _liquid_colour

TIER_NAMES = ("bottom", "middle", "top")
MIN_ABOVE_M = 0.015           # above the plank surface (plank top / texture noise below this)
MAX_ABOVE_M = 0.130           # a flask is 9.5 cm; higher = the arm or the plank above
FLASK_MIN_HEIGHT_M = 0.082    # at/above this a vessel is a flask (9.5 cm), below it a beaker (7.0 cm)
MIN_HEIGHT_M, MAX_HEIGHT_M = 0.040, 0.115
MIN_COMPONENT_PX = 50
RADIUS_SHIFT_M = 0.012        # centroid of the visible near side -> axis, along the viewing direction

_GRID_CACHE = {}


def _pixel_grid(shape):
    if shape not in _GRID_CACHE:
        h, w = shape
        v, u = np.mgrid[0:h, 0:w]
        _GRID_CACHE[shape] = (u.astype(np.float64), v.astype(np.float64))
    return _GRID_CACHE[shape]


def world_points(depth, intrinsics, world_transform):
    """(H, W, 3) world coordinates of every pixel (distance_to_image_plane depth; camera looks down local -Z)."""
    u, v = _pixel_grid(depth.shape)
    d = np.asarray(depth, np.float64)
    xc = (u - intrinsics["cx"]) * d / intrinsics["fx"]
    yc = -(v - intrinsics["cy"]) * d / intrinsics["fy"]
    local = np.stack([xc, yc, -d], axis=-1)
    T = np.asarray(world_transform, np.float64)
    return local @ T[:3, :3].T + T[:3, 3]


def project(pt, intrinsics, world_transform):
    local = np.linalg.inv(np.asarray(world_transform, np.float64)) @ np.array([pt[0], pt[1], pt[2], 1.0])
    if local[2] >= 0:
        return None
    z = -local[2]
    return (local[0] / z * intrinsics["fx"] + intrinsics["cx"], -local[1] / z * intrinsics["fy"] + intrinsics["cy"])


def detect_rack_vessels(rgb, depth, intrinsics, world_transform, shelf):
    """shelf: dict(tops=[...], y0s=[...], y1=float, x0=float, x1=float). Returns a list of detections:
    {key, kind, liquid, tier, tier_name, pos (base centre, world), bbox_px, height_m, n_px}."""
    if depth is None or rgb is None:
        return []
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    P = world_points(depth, intrinsics, world_transform)
    valid = np.isfinite(depth) & (depth > 0.05)
    cam_xy = np.asarray(world_transform, np.float64)[:2, 3]
    out = []
    for tier, (top, y0) in enumerate(zip(shelf["tops"], shelf["y0s"])):
        z = P[..., 2]
        m = (valid
             & (P[..., 0] > shelf["x0"] + 0.01) & (P[..., 0] < shelf["x1"] - 0.01)
             & (P[..., 1] > y0 - 0.005) & (P[..., 1] < shelf["y1"] - 0.03)
             & (z > top + MIN_ABOVE_M) & (z < top + MAX_ABOVE_M))
        if not m.any():
            continue
        mask = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lab in range(1, n_lab):
            x, y, w, h, area = stats[lab]
            if area < MIN_COMPONENT_PX:
                continue
            comp = (labels == lab) & m
            pts = P[comp]
            if len(pts) < MIN_COMPONENT_PX // 2:
                continue
            height = float(np.percentile(pts[:, 2], 98)) - top
            if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M):
                continue
            kind = "flask" if height >= FLASK_MIN_HEIGHT_M else "beaker"
            c_xy = pts[:, :2].mean(axis=0)
            away = c_xy - cam_xy
            away /= max(float(np.linalg.norm(away)), 1e-6)
            pos_xy = c_xy + away * RADIUS_SHIFT_M
            x0b, y0b, x1b, y1b = int(x), int(y), int(x + w - 1), int(y + h - 1)
            liquid = _liquid_colour(rgb, labels == lab, (x0b, y0b, x1b, y1b))
            key = VESSEL_KEYS.get((kind, liquid))
            out.append({"key": key, "kind": kind, "liquid": liquid, "tier": tier, "tier_name": TIER_NAMES[tier],
                        "pos": [float(pos_xy[0]), float(pos_xy[1]), float(top)],
                        "bbox_px": [x0b - 3, y0b - 3, x1b + 3, y1b + 3], "height_m": height, "n_px": int(area)})
    return out
