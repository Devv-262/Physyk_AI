"""PerceptionProvider — the shadow/stage/vision mode switch described in
cosmos_integration.md's Step 5. In shadow mode (the default here), this NEVER changes what
pose drives the robot: get_object_pose() always returns the stage (ground-truth USD) pose.
Cosmos's own estimate is still computed and compared, purely so a real delta can be logged
and exposed via snapshot() — this is how the doc describes shadow mode's entire purpose:
"it just starts populating perception.snapshot()."

Every failure path (timeout, bad JSON, no depth, object not visible, deprojected point
outside the workspace, delta too large in vision mode) falls back to the stage pose. There
is no retry — a single grounding call per get_object_pose(), matching the doc's
max_retries=0 guard rail; a slow/dead Cosmos degrades to pure stage behavior, not a hang.
"""

import time

import numpy as np

from . import cosmos_client, deproject

VALID_MODES = ("stage", "shadow", "vision")


class PerceptionProvider:
    def __init__(self, get_rgb, get_depth, intrinsics, world_transform, stage_lookup,
                 label_map, mode="shadow", workspace_aabb=None, sanity_gate_m=0.08,
                 patch=11, crop_aabb=None):
        """
        get_rgb / get_depth: zero-arg callables returning the latest RGB (H,W,3 uint8) and
            depth (H,W float32, distance_to_image_plane) frames from the overview camera.
        intrinsics: dict with fx, fy, cx, cy, width, height (SCENE_CAM_INTRINSICS).
        world_transform: 4x4 np.ndarray, camera-to-world (SCENE_CAM_WORLD_TRANSFORM).
        stage_lookup: callable(obj_key) -> np.ndarray world position, the existing real
            ground-truth query (get_live_object_pos).
        label_map: dict {obj_key: cosmos_label} e.g. {"red_cube": "red cube", ...} — the
            vocabulary Cosmos is asked about, mapped back to this project's own object keys.
        workspace_aabb: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) — a deprojected point outside
            this box is treated as a bad estimate and discarded, regardless of mode. This is
            deliberately a generous safety box, not the tight table region — see crop_aabb.
        sanity_gate_m: in vision mode only, if the Cosmos-estimated pose is farther than this
            from the stage pose, fall back to stage silently (logged in the snapshot, but the
            robot still gets the safe/ground-truth pose).
        crop_aabb: a TIGHT box around the actual table/workspace (same shape as
            workspace_aabb) used to crop the frame before sending it to Cosmos. Confirmed
            live to matter: the full overview-camera frame renders each cube as only ~15px
            across (the camera is intentionally wide/cinematic), and a Cosmos grounding
            error of a few pixels on an object that small deprojects into 10-20cm of
            real-world error — comfortably enough to blow the sanity gate on every call.
            Cropping to just the table roughly triples the cubes' pixel footprint, which is
            the actual fix; the AABB rejection above is a safety net, not a substitute for
            this. Falls back to no cropping (whole frame) if omitted or if projecting its
            corners fails for any reason.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"invalid perception mode {mode!r}, must be one of {VALID_MODES}")
        self.mode = mode
        self.get_rgb = get_rgb
        self.get_depth = get_depth
        self.intrinsics = intrinsics
        self.world_transform = world_transform
        self.stage_lookup = stage_lookup
        self.label_map = label_map
        self.workspace_aabb = workspace_aabb
        self.sanity_gate_m = sanity_gate_m
        self.patch = patch
        self._crop_rect = self._compute_crop_rect(crop_aabb) if crop_aabb is not None else None
        self._last_snapshot = {
            "mode": self.mode, "updated": False, "timestamp": None,
            "latency_ms": None, "error": None, "objects": {},
        }

    def _compute_crop_rect(self, crop_aabb):
        """Projects the 8 corners of crop_aabb into pixel space and returns a padded,
        clamped (u0, v0, u1, v1) integer crop rect, or None if that ever fails — a failure
        here just means grounding runs on the full frame (worse accuracy, never a crash)."""
        try:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = crop_aabb
            corners = [np.array([x, y, z]) for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)]
            pixels = [deproject.project_world_to_pixel(c, self.intrinsics, self.world_transform) for c in corners]
            pixels = [p for p in pixels if p is not None]
            if not pixels:
                return None
            us = [p[0] for p in pixels]
            vs = [p[1] for p in pixels]
            u0, u1 = min(us), max(us)
            v0, v1 = min(vs), max(vs)
            pad_u, pad_v = (u1 - u0) * 0.15, (v1 - v0) * 0.15
            width, height = self.intrinsics["width"], self.intrinsics["height"]
            u0 = max(0, int(u0 - pad_u))
            v0 = max(0, int(v0 - pad_v))
            u1 = min(width, int(u1 + pad_u))
            v1 = min(height, int(v1 + pad_v))
            if u1 - u0 < 20 or v1 - v0 < 20:
                return None
            return (u0, v0, u1, v1)
        except Exception:
            return None

    def _in_workspace(self, pt: np.ndarray) -> bool:
        if self.workspace_aabb is None or pt is None:
            return pt is not None
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.workspace_aabb
        return (xmin <= pt[0] <= xmax) and (ymin <= pt[1] <= ymax) and (zmin <= pt[2] <= zmax)

    def refresh(self, obj_keys):
        """Grounds ALL of obj_keys in one Cosmos call, deprojects each, compares to the
        real stage pose, and updates the internal snapshot. Does NOT return anything used
        for robot motion — callers that need a pose still call stage_lookup directly (or
        get_object_pose below, which in shadow/stage mode is equivalent to stage_lookup)."""
        if self.mode == "stage":
            # Doc's own instant-rollback mode: don't even call Cosmos.
            objects = {}
            for key in obj_keys:
                stage_pos = self.stage_lookup(key)
                objects[key] = {
                    "stage_pos_m": [float(v) for v in stage_pos],
                    "vision_pos_m": None, "delta_mm": None, "used": "stage",
                }
            self._last_snapshot = {
                "mode": self.mode, "updated": True, "timestamp": time.time(),
                "latency_ms": None, "error": None, "objects": objects,
            }
            return

        labels = [self.label_map[k] for k in obj_keys if k in self.label_map]
        try:
            rgb = self.get_rgb()
            depth = self.get_depth()
        except Exception as e:
            self._last_snapshot = {
                "mode": self.mode, "updated": False, "timestamp": time.time(),
                "latency_ms": None, "error": f"camera read failed: {e}", "objects": {},
            }
            return
        if rgb is None or depth is None or rgb.size == 0 or depth.size == 0:
            self._last_snapshot = {
                "mode": self.mode, "updated": False, "timestamp": time.time(),
                "latency_ms": None, "error": "no camera frame available yet", "objects": {},
            }
            return

        # Crop to the known table region before grounding (see __init__'s crop_aabb
        # docstring for why this matters) — points come back in the CROPPED image's own
        # pixel space, so offset by the crop origin before using them against the
        # full-frame depth array below.
        crop_u0, crop_v0 = 0, 0
        ground_rgb = rgb
        if self._crop_rect is not None:
            u0, v0, u1, v1 = self._crop_rect
            cropped = rgb[v0:v1, u0:u1]
            if cropped.size > 0:
                ground_rgb = cropped
                crop_u0, crop_v0 = u0, v0

        points, latency_ms, err = cosmos_client.ground_objects(ground_rgb, labels)

        objects = {}
        for key in obj_keys:
            stage_pos = self.stage_lookup(key)
            label = self.label_map.get(key)
            pixel = points.get(label) if label else None
            vision_pos = None
            if pixel is not None:
                full_frame_pixel = (pixel[0] + crop_u0, pixel[1] + crop_v0)
                dp = deproject.deproject_pixel(
                    full_frame_pixel[0], full_frame_pixel[1], depth, self.intrinsics,
                    self.world_transform, patch=self.patch,
                )
                if dp is not None and self._in_workspace(dp):
                    vision_pos = dp

            delta_mm = None
            if vision_pos is not None:
                delta_mm = float(np.linalg.norm(vision_pos - stage_pos) * 1000.0)

            objects[key] = {
                "stage_pos_m": [float(v) for v in stage_pos],
                "vision_pos_m": [float(v) for v in vision_pos] if vision_pos is not None else None,
                "delta_mm": delta_mm,
                "used": "stage",  # shadow/stage NEVER hand vision_pos to the robot
            }

        self._last_snapshot = {
            "mode": self.mode, "updated": True, "timestamp": time.time(),
            "latency_ms": latency_ms, "error": err, "objects": objects,
        }

    def get_object_pose(self, obj_key: str) -> np.ndarray:
        """The pose that should actually drive the robot for obj_key. In stage/shadow mode
        this is ALWAYS the stage (ground-truth) pose — matching the doc's own description
        of shadow mode verbatim. In vision mode, uses the last-refreshed Cosmos estimate IF
        it passed the workspace check and is within sanity_gate_m of stage; otherwise falls
        back to stage automatically."""
        stage_pos = self.stage_lookup(obj_key)
        if self.mode != "vision":
            return stage_pos
        entry = self._last_snapshot.get("objects", {}).get(obj_key)
        if not entry or entry.get("vision_pos_m") is None:
            return stage_pos
        vision_pos = np.array(entry["vision_pos_m"])
        if entry.get("delta_mm") is not None and entry["delta_mm"] > self.sanity_gate_m * 1000.0:
            return stage_pos  # sanity gate failed — silently fall back, per doc
        entry["used"] = "vision"
        return vision_pos

    def snapshot(self) -> dict:
        return self._last_snapshot
