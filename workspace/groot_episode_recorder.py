"""
Fine-tuning episode recorder for the Physyk cube/tray scene.

Purely additive: only used by isaac_sim_service.py when RECORD_EPISODES=1 is set in the
environment. When that flag is unset (the default), EpisodeRecorder.enabled is False and
every method is a cheap no-op — the existing pipeline's behavior is completely unchanged.

Runs INSIDE the Isaac Sim container's own Python (bundled Kit Python), which does NOT have
pyarrow/pandas installed — so this writes a dependency-free "raw" per-episode format (cv2 for
video, json for per-frame state/action) rather than LeRobot's parquet directly. A separate
offline script (convert_raw_episodes_to_lerobot.py, run under the Isaac-GR00T repo's own venv
which does have pyarrow/pandas) converts these raw episodes into the final GR00T-flavored
LeRobot v2 dataset. See physyk/workspace/scripts/generate_finetune_episodes.py for the driver
that triggers episodes end-to-end and physyk/workspace/scripts/convert_raw_episodes_to_lerobot.py
for the conversion step.

Raw per-episode layout written here:
    <output_dir>/episode_XXXXXX/
        scene.mp4       # raw (pre-HUD-overlay) scene/front camera, native sim resolution
        wrist.mp4        # raw wrist camera, native sim resolution
        frames.jsonl      # one JSON object per captured frame: {t, state} — action is derived
                          #   from consecutive states by convert_raw_episodes_to_lerobot.py
        meta.json         # {instruction, success, num_frames, fps}
"""

import json
import os
import time

import cv2
import numpy as np

# Fixed downward grasp orientation — Physyk has no real measured EE-orientation telemetry
# channel (see isaac_sim_service.py's own VLA_DOWNWARD_QUAT comment, ~line 966-971); the
# existing GR00T VLA integration already assumes this same constant for state feedback, so
# reusing it here for recorded episodes is consistent with established precedent, not a new
# approximation introduced by this recorder.
FIXED_ROLL_PITCH_YAW = (0.0, np.pi, 0.0)

# Capture cadence — matches demo_data/libero_demo's 20fps, independent of the sim's own
# internal physics/render rate (fps_counter in isaac_sim_service.py runs much higher).
RECORD_FPS = 20.0


class EpisodeRecorder:
    def __init__(self, output_dir: str):
        self.enabled = os.environ.get("RECORD_EPISODES", "0") == "1"
        self.output_dir = output_dir
        self._episode_idx = 0
        self._active = None  # dict while an episode is being recorded, else None
        self._last_frame_t = 0.0
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            # Resume numbering after whatever's already on disk, so re-running the driver
            # script doesn't clobber a previous batch's episodes.
            existing = [d for d in os.listdir(self.output_dir) if d.startswith("episode_")]
            self._episode_idx = len(existing)
            print(f"[EpisodeRecorder] ENABLED — writing to {self.output_dir}, "
                  f"resuming at episode index {self._episode_idx}", flush=True)

    def start_episode(self, instruction: str):
        if not self.enabled or self._active is not None:
            return
        self._active = {
            "instruction": instruction,
            "start_time": time.time(),
            "frames": [],       # list of {"t": float, "state": [8 floats], "action": [7 floats]}
            "scene_writer": None,
            "wrist_writer": None,
            "scene_path": None,
            "wrist_path": None,
        }
        self._last_frame_t = 0.0

    def log_frame(self, raw_scene_rgb: np.ndarray, raw_wrist_rgb: np.ndarray,
                  ee_pos, gripper_width: float) -> None:
        """Call once per sim frame while a pick_place task is active. Internally throttles to
        RECORD_FPS so callers don't need to rate-limit themselves.

        Only proprioceptive STATE is logged here (ee_pos + gripper) — GR00T's per-frame
        ACTION (the delta needed to reach the next state) is derived afterwards, from
        consecutive states, by the offline conversion script. This keeps the in-sim hook
        simple: it doesn't need access to the controller's internal joint-space command,
        only what's already in sim_telemetry."""
        if not self.enabled or self._active is None:
            return
        now = time.time() - self._active["start_time"]
        if now - self._last_frame_t < (1.0 / RECORD_FPS):
            return
        self._last_frame_t = now

        ep = self._active
        if ep["scene_writer"] is None:
            ep_dir = os.path.join(self.output_dir, f"episode_{self._episode_idx:06d}")
            os.makedirs(ep_dir, exist_ok=True)
            h, w = raw_scene_rgb.shape[:2]
            hw_, ww_ = raw_wrist_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            ep["scene_path"] = os.path.join(ep_dir, "scene.mp4")
            ep["wrist_path"] = os.path.join(ep_dir, "wrist.mp4")
            ep["frames_path"] = os.path.join(ep_dir, "frames.jsonl")
            ep["meta_path"] = os.path.join(ep_dir, "meta.json")
            ep["scene_writer"] = cv2.VideoWriter(ep["scene_path"], fourcc, RECORD_FPS, (w, h))
            ep["wrist_writer"] = cv2.VideoWriter(ep["wrist_path"], fourcc, RECORD_FPS, (ww_, hw_))
            ep["frames_file"] = open(ep["frames_path"], "w")

        # Annotator frames come out RGB(A); VideoWriter expects BGR.
        scene_bgr = cv2.cvtColor(np.asarray(raw_scene_rgb)[..., :3], cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(np.asarray(raw_wrist_rgb)[..., :3], cv2.COLOR_RGB2BGR)
        ep["scene_writer"].write(scene_bgr)
        ep["wrist_writer"].write(wrist_bgr)

        roll, pitch, yaw = FIXED_ROLL_PITCH_YAW
        state = [
            float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]),
            roll, pitch, yaw,
            gripper_width / 2.0, gripper_width / 2.0,
        ]
        ep["frames_file"].write(json.dumps({"t": now, "state": state}) + "\n")
        ep["frames"].append(True)  # just used as a frame counter
        ep["last_t"] = now

    def end_episode(self, success: bool) -> None:
        """Flush the episode to disk if successful; discard (delete partial files) if not."""
        if not self.enabled or self._active is None:
            return
        ep = self._active
        self._active = None
        num_frames = len(ep["frames"])
        if ep["scene_writer"] is not None:
            ep["scene_writer"].release()
            ep["wrist_writer"].release()
            ep["frames_file"].close()

        if not success or num_frames == 0:
            # Drop failed/empty episodes rather than write them — keeps the dataset clean
            # without a separate filtering pass later.
            for p in (ep.get("scene_path"), ep.get("wrist_path"), ep.get("frames_path"), ep.get("meta_path")):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            ep_dir = os.path.dirname(ep["scene_path"]) if ep.get("scene_path") else None
            if ep_dir and os.path.isdir(ep_dir) and not os.listdir(ep_dir):
                os.rmdir(ep_dir)
            print(f"[EpisodeRecorder] Episode DROPPED (success={success}, frames={num_frames})", flush=True)
            return

        # The raw scene.mp4/wrist.mp4 were written with cv2.VideoWriter fixed at RECORD_FPS
        # (20), but the sim's actual achievable render rate is much lower (~7fps observed) —
        # frames arrive slower than the throttle interval, so every real frame gets captured,
        # but the file's declared playback speed would be wrong if taken at face value. The
        # ACTUAL measured fps (frame count / real elapsed episode time) is recorded here so
        # convert_raw_episodes_to_lerobot.py can re-encode at the true rate instead of
        # inheriting this recorder's placeholder constant.
        measured_fps = round(num_frames / ep["last_t"], 3) if ep.get("last_t", 0) > 0 else RECORD_FPS
        with open(ep["meta_path"], "w") as f:
            json.dump({
                "instruction": ep["instruction"],
                "success": True,
                "num_frames": num_frames,
                "fps": measured_fps,
            }, f, indent=2)
        print(f"[EpisodeRecorder] Episode {self._episode_idx:06d} SAVED "
              f"({num_frames} frames, '{ep['instruction']}')", flush=True)
        self._episode_idx += 1
