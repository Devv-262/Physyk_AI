"""Glassware (Florence flasks, beakers) detector for the fixed overview camera.

Colour alone can't find glass: the vessels' liquids share hues with the red/green/blue cubes, and an EMPTY
flask has no colour at all. But the depth camera DOES see the glass shell, so vessels are found by GEOMETRY
and only then named by colour:

  1. Deproject the table region with depth, subtract the (fixed) table plane -> height above the table.
  2. Candidate pixels = a band of heights that includes vessels but excludes the table, the tray slab,
     the robot (taller); cluster them in the image (connected components). Clusters that fail the shape test
     are re-tried after cutting them along depth discontinuities, which separates vessels that merely touch or
     overlap in the camera view.
  3. Per cluster, measure the TOP height and the width of its top slice:
        flask  : top ~9.5 cm, top slice = the narrow neck (<~3.5 cm wide)
        beaker : top ~7.0 cm, top slice = the full 5 cm rim
        cube   : top ~5.0 cm (stacks 10 / 15 cm) -> NOT a vessel, left to the cube detector
  4. Name it by its liquid colour (vivid pixels inside the cluster): blue / red / green, or "empty" if none.
  5. Position = centre of the top ring (the rim/neck is a full circle seen from above, so its mean is the axis);
     base height = top - known vessel height.

Pure numpy/cv2 — no simulator imports — so it is testable on saved frames.
"""

import cv2
import numpy as np

# Registry key for each (kind, liquid) combination.
VESSEL_KEYS = {
    ("flask", "blue"): "flask_blue",
    ("flask", "red"): "flask_red",
    ("flask", "empty"): "flask_empty",
    ("beaker", "red"): "beaker_red",
    ("beaker", "green"): "beaker_green",
}
VESSEL_HEIGHT_M = {"flask": 0.095, "beaker": 0.070}          # known glass heights (STL dimensions)

MIN_HEIGHT_M = 0.020          # below this = table texture/shadow noise or the tray slab
MAX_HEIGHT_M = 0.125          # above this = robot / cube stacks, never a single vessel
DEPTH_JUMP_M = 0.012          # an adjacent-pixel depth step bigger than this is an occlusion edge between objects
MAX_SINGLE_PX = 650           # a lone vessel covers ~370-520 px at this camera distance; bigger = several blended
MIN_CLUSTER_PX = 12
TOP_SLICE_M = 0.007           # points within this of the top form the "top ring"

# Liquid hues (OpenCV H in [0,179]), measured from live frames: blue ~104, red ~0, green ~73. Vivid = S, V high.
LIQUID_HUES = {"red": [(0, 13), (172, 179)], "green": [(64, 89)], "blue": [(94, 124)]}
LIQUID_S_MIN, LIQUID_V_MIN = 95, 90
MIN_LIQUID_PX = 6


def _liquid_colour(rgb, cluster_mask, bbox):
    """Dominant vivid liquid colour inside the cluster, or 'empty' if there are (almost) no vivid pixels."""
    x0, y0, x1, y1 = bbox
    rgb, cluster_mask = rgb[y0:y1 + 1, x0:x1 + 1], cluster_mask[y0:y1 + 1, x0:x1 + 1]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2HSV)
    vivid = cluster_mask & (hsv[:, :, 1] >= LIQUID_S_MIN) & (hsv[:, :, 2] >= LIQUID_V_MIN)
    best, best_n = "empty", MIN_LIQUID_PX - 1
    for name, ranges in LIQUID_HUES.items():
        sel = np.zeros_like(vivid)
        for lo, hi in ranges:
            sel |= (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
        n = int((vivid & sel).sum())
        if n > best_n:
            best, best_n = name, n
    return best


_TABLE_RAY_CACHE = {}


def _table_ray_maps(intrinsics, world_transform, shape):
    """Per-pixel (depth to the table plane Z=0 along that pixel's ray, and the ray's world Z component),
    cached: the camera never moves. A point at depth t on that ray has height  h = Dz * (t - Dtab) above the
    table, so a full-image height map needs no 3D deprojection."""
    key = (id(world_transform), shape, round(float(intrinsics["fx"]), 3))
    hit = _TABLE_RAY_CACHE.get(key)
    if hit is not None:
        return hit
    Hh, Ww = shape
    vv, uu = np.mgrid[0:Hh, 0:Ww]
    d_local = np.stack([(uu - intrinsics["cx"]) / intrinsics["fx"],
                        -(vv - intrinsics["cy"]) / intrinsics["fy"],
                        -np.ones((Hh, Ww))], axis=-1)
    R = np.asarray(world_transform)[:3, :3]
    eye_z = float(np.asarray(world_transform)[2, 3])
    dz = d_local @ R[2, :]                       # world Z component of each pixel's (z-depth-normalised) ray
    with np.errstate(divide="ignore", invalid="ignore"):
        dtab = np.where(dz < -1e-6, -eye_z / dz, np.nan)
    maps = (dtab.astype(np.float32), dz.astype(np.float32))
    _TABLE_RAY_CACHE.clear()
    _TABLE_RAY_CACHE[key] = maps
    return maps


def detect_vessels(rgb, depth, intrinsics, world_transform, roi, plane=None, debug=None):
    """Returns (found, exclusion_mask).
    found: {registry_key: {position, yaw_deg, mask_px, top_px, centroid_px, bbox_px, kind, liquid, height_m}}
    exclusion_mask: uint8 HxW (255 on vessel pixels, dilated) — pass it to the cube detector so a red liquid is
    never mistaken for the red cube.
    plane: optional table-plane fit {"a","b","c"} (Z = a + bX + cY); only its offset `a` is used here.
    debug: optional list; if given, one dict per candidate cluster explaining the decision is appended."""
    from .color_segment import _deproject_pixels
    Hh, Ww = depth.shape[:2]
    exclusion = np.zeros((Hh, Ww), dtype=np.uint8)
    dtab, dz = _table_ray_maps(intrinsics, world_transform, (Hh, Ww))
    valid = (roi > 0) & np.isfinite(depth) & (depth > 0) & np.isfinite(dtab)
    hfull = np.where(valid, dz * (depth - dtab), np.nan)          # height above Z=0 for every pixel
    if plane:
        hfull = hfull - float(plane["a"])
    coarse = np.isfinite(hfull) & (hfull >= MIN_HEIGHT_M) & (hfull <= MAX_HEIGHT_M)
    vs, us = np.nonzero(coarse)
    if len(us) == 0:
        return {}, exclusion
    # exact XY only for the few thousand candidate pixels
    pts = _deproject_pixels(us.astype(np.float64), vs.astype(np.float64), depth[vs, us].astype(np.float64),
                            intrinsics, world_transform)
    height = hfull[vs, us].astype(np.float64)
    cand_img = np.zeros((Hh, Ww), dtype=np.uint8)
    cand_img[vs, us] = 255
    cand_img = cv2.morphologyEx(cand_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_img, connectivity=8)

    hmap = np.full((Hh, Ww), np.nan, dtype=np.float32)
    xmap = np.full((Hh, Ww), np.nan, dtype=np.float32)
    ymap = np.full((Hh, Ww), np.nan, dtype=np.float32)
    hmap[vs, us], xmap[vs, us], ymap[vs, us] = height, pts[:, 0], pts[:, 1]

    def classify(cmask, dbg_tag=""):
        """Shape + colour test for one cluster. Returns (key, entry) or None."""
        h = hmap[cmask]
        good = np.isfinite(h)
        if int(good.sum()) < MIN_CLUSTER_PX:
            return None
        h_top = float(np.percentile(h[good], 98))
        in_top = cmask & (hmap >= h_top - TOP_SLICE_M)
        tx, ty = xmap[in_top], ymap[in_top]
        if len(tx) < 3:
            return None
        top_w = float(max(tx.max() - tx.min(), ty.max() - ty.min()))   # top-slice extent (m)
        vs_c, us_c = np.nonzero(cmask)
        bbox = [int(us_c.min()), int(vs_c.min()), int(us_c.max()), int(vs_c.max())]
        dbg = None
        if debug is not None:
            dbg = {"tag": dbg_tag, "px": int(cmask.sum()), "centroid_px": (round(float(us_c.mean())), round(float(vs_c.mean()))),
                   "h_top_cm": round(h_top * 100, 1), "top_w_cm": round(top_w * 100, 1),
                   "xy": (round(float(tx.mean()), 3), round(float(ty.mean()), 3))}
            debug.append(dbg)
        if 0.085 <= h_top <= 0.108 and top_w <= 0.040:
            kind = "flask"          # tall, narrow neck on top
        elif 0.058 <= h_top <= 0.082 and top_w >= 0.038:
            kind = "beaker"         # full-width rim at 7 cm
        else:
            if dbg is not None:
                dbg["verdict"] = "rejected: not a flask/beaker profile"
            return None             # merged blobs, the robot, or noise
        liquid = _liquid_colour(rgb, cmask, bbox)
        key = VESSEL_KEYS.get((kind, liquid))
        if dbg is not None:
            dbg.update(kind=kind, liquid=liquid, key=key, verdict="accepted" if key else "rejected: no registry key")
        if key is None:
            return None
        return key, {
            "position": [float(tx.mean()), float(ty.mean()), float(h_top - VESSEL_HEIGHT_M[kind])],
            "yaw_deg": 0.0,
            "mask_px": int(cmask.sum()),
            "top_px": int(in_top.sum()),
            "centroid_px": [float(us_c.mean()), float(vs_c.mean())],
            "bbox_px": bbox,
            "kind": kind, "liquid": liquid, "height_m": h_top,
        }

    # depth discontinuities (occlusion edges between different objects), computed once, lazily
    edge = None

    found = {}
    cluster_masks = []

    def accept(res, cmask):
        if res is None:
            return False
        key, entry = res
        if key not in found or entry["mask_px"] > found[key]["mask_px"]:   # same identity twice: keep the larger
            found[key] = entry
        cluster_masks.append(cmask)
        return True

    def near(a_mask, b_mask):
        return bool((cv2.dilate(a_mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & b_mask).any())

    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < MIN_CLUSTER_PX:
            continue
        cmask = labels == lab
        if area <= MAX_SINGLE_PX and accept(classify(cmask), cmask):
            continue
        # Failed the shape test, or too big to be one vessel (several blended together in the camera view): cut
        # the blob along depth jumps (occlusion edges between objects) and test the pieces separately.
        if area < 2 * MIN_CLUSTER_PX:
            continue
        if edge is None:
            d = np.where(np.isfinite(depth), depth, 0.0)
            edge = np.zeros((Hh, Ww), dtype=bool)
            jx = np.abs(np.diff(d, axis=1)) > DEPTH_JUMP_M
            jy = np.abs(np.diff(d, axis=0)) > DEPTH_JUMP_M
            edge[:, :-1] |= jx
            edge[:, 1:] |= jx
            edge[:-1, :] |= jy
            edge[1:, :] |= jy
        piece_img = (cmask & ~edge).astype(np.uint8) * 255
        pn, plabels, pstats, _ = cv2.connectedComponentsWithStats(piece_img, connectivity=8)
        rejected = []
        accepted_here = np.zeros((Hh, Ww), dtype=bool)
        for pl in range(1, pn):
            if pstats[pl, cv2.CC_STAT_AREA] >= MIN_CLUSTER_PX:
                pm = plabels == pl
                if accept(classify(pm, "split-piece"), pm):
                    accepted_here |= pm
                else:
                    rejected.append(pm)
        # A vessel's own rim / far wall can also contain depth jumps, which cuts ONE vessel into several pieces
        # that each fail the shape test. Re-join neighbouring rejected pieces (re-including the cut pixels
        # between them) and test the union.
        used = set()
        for i in range(len(rejected)):
            for j in range(i + 1, len(rejected)):
                if i in used or j in used or not near(rejected[i], rejected[j]):
                    continue
                union = cv2.dilate((rejected[i] | rejected[j]).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                union &= cmask & ~cv2.dilate(accepted_here.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                if accept(classify(union, "rejoined-pieces"), union):
                    used.update((i, j))

    for cm in cluster_masks:
        exclusion[cm] = 255
    if found:
        exclusion = cv2.dilate(exclusion, np.ones((9, 9), np.uint8))
    return found, exclusion
