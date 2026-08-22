"""Pixel + depth -> world-space deprojection for the fixed overview camera plumbed in
isaac_sim_service.py (Step 3 of cosmos_integration.md). Standard pinhole math plus the
camera's known world transform — no live API calls, everything here is pure numpy given
values already computed/measured elsewhere.
"""

import numpy as np


def _median_depth_patch(depth_frame: np.ndarray, u: float, v: float, patch: int = 11):
    """Median depth (meters) over a patch x patch window centered on (u, v), ignoring
    non-finite/zero samples — avoids picking up a single noisy or background/edge pixel,
    per cosmos_integration.md's own guard-rail recommendation."""
    h, w = depth_frame.shape[0], depth_frame.shape[1]
    cu, cv = int(round(u)), int(round(v))
    half = patch // 2
    u0, u1 = max(0, cu - half), min(w, cu + half + 1)
    v0, v1 = max(0, cv - half), min(h, cv + half + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    window = depth_frame[v0:v1, u0:u1].astype(np.float64)
    finite = window[np.isfinite(window) & (window > 0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def project_world_to_pixel(world_pt, intrinsics: dict, world_transform: np.ndarray):
    """Inverse of deproject_pixel: world XYZ -> (u, v) pixel in this camera's real image, or
    None if the point is behind the camera. Used to compute a real crop region around the
    known physical workspace before grounding (see provider.py) — not for anything that
    drives the robot."""
    world_transform = np.asarray(world_transform)
    inv = np.linalg.inv(world_transform)
    local = inv @ np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
    x_local, y_local, z_local = local[0], local[1], local[2]
    if z_local >= 0:
        return None  # behind the camera (camera looks down local -Z, see deproject_pixel)
    z = -z_local
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    u = x_local / z * fx + cx
    v = -y_local / z * fy + cy  # image v increases downward, camera-local y increases upward
    return (u, v)


def deproject_pixel(u: float, v: float, depth_frame: np.ndarray, intrinsics: dict,
                     world_transform: np.ndarray, patch: int = 11):
    """Returns a world-space np.ndarray([x, y, z]) or None if depth at this pixel is
    unusable. `depth_frame` must be distance_to_image_plane (z-depth along the optical
    axis), matching the annotator this project uses — NOT distance_to_camera.

    Camera convention (USD/Replicator, confirmed by this project's own extrinsics
    construction in isaac_sim_service.py): the camera looks down its local -Z axis, local
    +X is right, local +Y is up. Image pixel v increases downward, opposite of local +Y, so
    the y component is negated relative to the raw pixel offset.
    """
    z = _median_depth_patch(depth_frame, u, v, patch=patch)
    if z is None:
        return None
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x_local = (u - cx) / fx * z
    y_local = -(v - cy) / fy * z
    z_local = -z  # points in front of the camera sit at negative local Z, see docstring
    local_pt = np.array([x_local, y_local, z_local, 1.0])
    world_pt = np.asarray(world_transform) @ local_pt
    return world_pt[:3]
