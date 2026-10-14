"""Color-mask cube detector for the fixed overview camera — the "pure perception" localizer.

The scene is three saturated cubes (red / green / blue) on a dark table, so a hue threshold
gives an exact per-cube mask in a few ms with no GPU and no model. Pipeline per cube:
  RGB -> HSV hue mask -> largest blob -> deproject every mask pixel with the depth frame ->
  keep the TOP face (the camera is oblique, so the mask also covers side faces) ->
  top-face XY centroid = cube center XY, top Z - half height = cube center Z; the min-area
  rectangle of the top face gives yaw (reported only; the grasp stays top-down).

Pure numpy/cv2 — no simulator imports — so it is testable on saved frames.
"""

import cv2
import numpy as np

# OpenCV HSV: H in [0,179], S/V in [0,255]. Thresholds are set from MEASURED pixels of the live
# overview frame (studio lighting washes the cubes out): red ~H7/S116/V219, green ~H76/S106/V224,
# blue ~H99/S88/V223. The ground grid is H~110/V~150-170 (close to the blue cube's hue), so blue
# also needs a high V and the search is restricted to the table area (ROI) as well.
# "s_weak"/"v_weak" are the looser thresholds for the TOP face, which renders nearly white
# (measured S 40-56, V ~245 vs the sides' S 86-129): it can't pass s_min, so it is recovered by
# growing the strong (side-face) mask into same-hue, bright, slightly-colored neighbors. The bare
# table (S ~13) and the white robot (S ~0-10) stay below s_weak.
COLOR_SPECS = {
    "red_cube":   {"hue": [(0, 12), (170, 179)], "s_min": 80, "v_min": 150, "s_weak": 35, "v_weak": 200},
    "green_cube": {"hue": [(60, 90)],            "s_min": 70, "v_min": 150, "s_weak": 35, "v_weak": 200},
    "blue_cube":  {"hue": [(92, 108)],           "s_min": 65, "v_min": 190, "s_weak": 35, "v_weak": 215},
    # The yellow tray (a flat slab). It never moves and is a fixed pose in camera-only mode, so this spec is only used by
    # locate_cube() on the tray (the offline perception test); its top face renders pale (S ~65), its side faces S ~110.
    "target_tray": {"hue": [(20, 38)],           "s_min": 85,  "v_min": 150, "s_weak": 45, "v_weak": 200, "slab": True},
}

# Vertical half-extent of each object, to turn the measured TOP surface height into a center Z.
HALF_HEIGHT_M = {"red_cube": 0.025, "green_cube": 0.025, "blue_cube": 0.025, "target_tray": 0.015}

# World-space box the cubes can ever be in (table top, with room for a 3-cube stack). Its
# projection into the image is the only place we look — keeps the blue ground grid out.
TABLE_ROI_BOX = ((0.0, 0.9), (-0.65, 0.65), (-0.02, 0.22))

TOP_GROW_PX = 25              # how far (px) the top face can extend from the side faces (~1 cube)
MIN_MASK_PIXELS = 25          # a cube is ~20x20 px in the 960x540 overview frame
TOP_FACE_BAND_M = 0.002       # points within this of the top surface count as the top face
                              # (was 6 mm, which also swallowed a strip of side face and dragged
                              # the center toward the camera; sim depth is essentially noise-free)
CUBE_EDGE_SHRINK_M = 0.003    # the 1-px silhouette erosion trims ~3 mm off the visible extents
YAW_PRIOR_M_PER_DEG = 0.0003  # penalty per degree of yaw in the square fit (prefers upright)


def roi_mask(shape, intrinsics, world_transform, box=TABLE_ROI_BOX) -> np.ndarray:
    """uint8 HxW mask (255 inside) of the image region covered by the world-space `box`:
    convex hull of its 8 projected corners. Corners behind the camera are skipped."""
    from .deproject import project_world_to_pixel
    (x0, x1), (y0, y1), (z0, z1) = box
    pts = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                px = project_world_to_pixel((x, y, z), intrinsics, world_transform)
                if px is not None:
                    pts.append(px)
    roi = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(roi, hull, 255)
    else:
        roi[:] = 255
    return roi


def color_mask(rgb: np.ndarray, obj_key: str, roi: np.ndarray = None) -> np.ndarray:
    """Boolean HxW mask of pixels matching obj_key's color (inside `roi` if given), largest
    connected blob only (drops speckle and any stray same-colored pixels elsewhere)."""
    spec = COLOR_SPECS[obj_key]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in spec["hue"]:
        mask |= cv2.inRange(hsv, (lo, spec["s_min"], spec["v_min"]), (hi, 255, 255))
    if "s_weak" in spec and mask.any():
        # Grow into the top face: same hue, brighter but paler than the sides, and near the strong mask.
        weak = np.zeros_like(mask)
        for lo, hi in spec["hue"]:
            weak |= cv2.inRange(hsv, (lo, spec["s_weak"], spec["v_weak"]), (hi, 255, 255))
        near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOP_GROW_PX, TOP_GROW_PX)))
        mask |= cv2.bitwise_and(weak, near)
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _deproject_pixels(us, vs, zs, intrinsics, world_transform):
    """Vectorized pinhole deprojection (same convention as deproject.deproject_pixel: camera
    looks down local -Z, local +Y is up, image v grows downward). Returns Nx3 world points."""
    x_local = (us - intrinsics["cx"]) / intrinsics["fx"] * zs
    y_local = -(vs - intrinsics["cy"]) / intrinsics["fy"] * zs
    local = np.stack([x_local, y_local, -zs, np.ones_like(zs)], axis=0)
    return (np.asarray(world_transform) @ local)[:3].T


def locate_cube(rgb, depth, obj_key, intrinsics, world_transform, cube_half_height=None, roi=None):
    """Returns dict {position: [x,y,z], yaw_deg, mask_px, top_px, centroid_px, bbox_px} or None
    if the object isn't found / has no usable depth. Works for the cubes and the tray (see
    COLOR_SPECS / HALF_HEIGHT_M); cube_half_height overrides the per-object default."""
    half_h = HALF_HEIGHT_M.get(obj_key, 0.025) if cube_half_height is None else cube_half_height
    is_slab = bool(COLOR_SPECS[obj_key].get("slab"))
    mask = color_mask(rgb, obj_key, roi)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        return None
    vs_all, us_all = np.nonzero(mask)                      # full silhouette (for the overlay box)
    bbox = [int(us_all.min()), int(vs_all.min()), int(us_all.max()), int(vs_all.max())]

    # Shave the silhouette by one pixel before using depth: edge pixels can carry background depth.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    use = eroded if int(eroded.sum()) >= MIN_MASK_PIXELS else mask
    vs, us = np.nonzero(use)
    zs = depth[vs, us].astype(np.float64)
    ok = np.isfinite(zs) & (zs > 0)
    if int(ok.sum()) < MIN_MASK_PIXELS:
        return None
    us, vs, zs = us[ok].astype(np.float64), vs[ok].astype(np.float64), zs[ok]

    pts = _deproject_pixels(us, vs, zs, intrinsics, world_transform)
    top_z = np.percentile(pts[:, 2], 97)              # robust top surface height
    top = pts[pts[:, 2] >= top_z - TOP_FACE_BAND_M]
    if len(top) < 6:
        return None
    top_surface_z = float(np.median(top[:, 2]))

    if is_slab:
        # Tray: outline center of its top surface (cubes resting in it hide part of the yellow).
        (rcx, rcy), (_, _), angle = cv2.minAreaRect((top[:, :2] * 1000.0).astype(np.float32))
        cx, cy = rcx / 1000.0, rcy / 1000.0
        yaw_deg = float(((angle + 45.0) % 90.0) - 45.0)
    else:
        # Cube: fit a square of the KNOWN side length to the footprint of every visible point
        # (top face + the two camera-facing sides). Works even when the top face is hidden by a
        # cube stacked on it — the two side faces still form an "L" that pins the square down.
        # (Averaging top-face pixels is biased toward the camera: perspective gives the far
        # part of the face fewer pixels. Generic min-area-rect degenerates on an "L".)
        footprint = pts[pts[:, 2] > 0.006][:, :2]
        if len(footprint) < 6:
            return None
        (cx, cy), yaw_deg = _fit_square(footprint, 2.0 * half_h - CUBE_EDGE_SHRINK_M)

    return {
        "position": [float(cx), float(cy), top_surface_z - half_h],
        "yaw_deg": float(yaw_deg),
        "mask_px": int(mask.sum()),
        "top_px": int(len(top)),
        "centroid_px": [float(us_all.mean()), float(vs_all.mean())],
        "bbox_px": bbox,
    }


def fit_table_plane(rgb, depth, intrinsics, world_transform, roi, gate_m=0.03, iters=4, max_pts=30000):
    """Self-calibration from the one thing that never moves: the table top, a flat plane at Z = 0.

    Deprojects the bare-table pixels (gray: low saturation, not dark relative to the table, inside the table ROI) and
    fits Z = a + b*X + c*Y to them with iterative outlier rejection (drops the robot's white links,
    shadows-edge pixels, anything not on the plane). If the camera pose/intrinsics were perfect the
    fit would be exactly Z = 0; whatever it isn't is the camera's vertical error, which the caller
    subtracts from every detection. Uses no simulator coordinates — only the known plane height.

    Returns {"a","b","c","n","rms_mm"} or None if too few table pixels were found."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2HSV)
    low_sat = (hsv[:, :, 1] < 40) & (roi > 0) & np.isfinite(depth) & (depth > 0)
    # A fixed brightness floor (V > 120) only works under bright lighting: in a darker room (ceiling, dimmer lights) the
    # table falls below it, only a patch of pixels survives, and the fitted plane tilts (cube heights then read 1-2 cm off).
    # The floor is relative to the table's own brightness instead; the depth gate and outlier rejection below still throw out
    # the robot, cubes and anything else that is not on the plane.
    v_all = hsv[:, :, 2][low_sat]
    v_floor = max(35.0, 0.5 * float(np.median(v_all))) if v_all.size else 120.0
    sel = low_sat & (hsv[:, :, 2] > v_floor)
    vs, us = np.nonzero(sel)
    if len(us) < 2000:
        return None
    if len(us) > max_pts:
        idx = np.random.default_rng(0).choice(len(us), max_pts, replace=False)
        vs, us = vs[idx], us[idx]
    pts = _deproject_pixels(us.astype(np.float64), vs.astype(np.float64),
                            depth[vs, us].astype(np.float64), intrinsics, world_transform)
    pts = pts[np.abs(pts[:, 2]) < gate_m]            # coarse gate around the expected plane Z = 0
    if len(pts) < 1000:
        return None
    keep = np.ones(len(pts), dtype=bool)
    coef = np.zeros(3)
    for _ in range(iters):
        A = np.column_stack([np.ones(keep.sum()), pts[keep, 0], pts[keep, 1]])
        coef, *_ = np.linalg.lstsq(A, pts[keep, 2], rcond=None)
        resid = pts[:, 2] - (coef[0] + coef[1] * pts[:, 0] + coef[2] * pts[:, 1])
        keep = np.abs(resid) < max(0.004, 2.5 * float(np.std(resid[keep])))
        if keep.sum() < 1000:
            return None
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]),
            "n": int(keep.sum()), "rms_mm": float(np.sqrt(np.mean(resid[keep] ** 2)) * 1000.0)}


def _fit_square(footprint, side, yaw_range_deg=45.0, step_deg=1.0, yaw_prior=YAW_PRIOR_M_PER_DEG):
    """Center and yaw of the `side` x `side` square that best explains a footprint (Nx2, meters):
    over candidate yaw angles, the frame whose bounding box is closest to side x side wins, with a
    small penalty on |yaw| (cubes here stay nearly upright, and the fit is weakly constrained)."""
    best = None
    for th in np.arange(-yaw_range_deg, yaw_range_deg, step_deg):
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        rot = np.array([[c, s], [-s, c]])
        q = footprint @ rot.T
        lo, hi = q.min(axis=0), q.max(axis=0)
        w, h = hi - lo
        loss = abs(w - side) + abs(h - side) + yaw_prior * abs(th)
        if best is None or loss < best[0]:
            best = (loss, th, rot, (lo + hi) / 2.0)
    _, th, rot, center_rot = best
    return rot.T @ center_rot, th
