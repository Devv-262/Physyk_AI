#!/usr/bin/env python3
"""
Convert raw episodes recorded by groot_episode_recorder.py (cv2 mp4 + frames.jsonl per
episode, written from inside the Isaac Sim container's dependency-light Python) into a
GR00T-flavored LeRobot v2 dataset (parquet + meta/*.json), matching the LIBERO_PANDA
embodiment schema exactly (see demo_data/libero_demo/meta/modality.json and info.json).

Run with the Isaac-GR00T repo's own venv (has pandas/pyarrow/opencv):
    cd /opt/dlami/nvme/physyk/workspace/Isaac-GR00T
    .venv/bin/python ../scripts/convert_raw_episodes_to_lerobot.py \
        --raw-dir /opt/dlami/nvme/physyk/workspace/finetune_data/raw_episodes \
        --out-dir /opt/dlami/nvme/physyk/workspace/finetune_data/physyk_cubes_40

Does not modify anything under Isaac-GR00T/ or touch the raw episodes (read-only on input).

NOTE: GR00T's dataset loader (gr00t/data/dataset/lerobot_episode_loader.py) decodes video via
torchcodec, which needs system FFmpeg shared libraries — this host initially lacked them
(torchcodec import failed: "Could not load libtorchcodec"); FFmpeg 4.4.2 has since been
installed via apt and torchcodec now imports successfully. Videos here are written with
cv2's mp4v codec (a widely-supported baseline, not the av1 the original checkpoint's own
demo data used) — fine for a from-scratch fine-tune dataset since the loader only needs to
decode it, not match the pretrain codec exactly.
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_SIZE = (256, 256)  # (w, h) — matches demo_data/libero_demo and the checkpoint's
                          # image_target_size in config.json


def resize_video(src_path: str, dst_path: str, fps: float) -> int:
    """Re-encode src_path to VIDEO_SIZE at dst_path, using the per-episode MEASURED fps (see
    groot_episode_recorder.py's end_episode) rather than the raw file's own declared rate —
    the raw recording was written with a fixed placeholder fps that doesn't match the sim's
    actual achieved render rate. Returns frame count."""
    cap = cv2.VideoCapture(src_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, VIDEO_SIZE)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    return n


def build_action_from_states(states: list[list[float]]) -> list[list[float]]:
    """action[i] = delta needed to go from state[i] to state[i+1]: xyz delta (roll/pitch/yaw
    fixed at 0 delta, since Physyk holds a constant downward orientation throughout every
    pick-place), gripper = absolute width at i+1 (matches LIBERO's own action convention,
    where gripper is commanded as an absolute target, not a delta). Last frame repeats the
    second-to-last action (no next state to diff against)."""
    actions = []
    for i in range(len(states)):
        j = min(i + 1, len(states) - 1)
        s_now, s_next = states[i], states[j]
        dx = s_next[0] - s_now[0]
        dy = s_next[1] - s_now[1]
        dz = s_next[2] - s_now[2]
        gripper_next = s_next[6] + s_next[7]  # undo the /2 split back to full width
        actions.append([dx, dy, dz, 0.0, 0.0, 0.0, gripper_next])
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--robot-type", default="franka")
    args = ap.parse_args()

    raw_episodes = sorted(
        d for d in os.listdir(args.raw_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(args.raw_dir, d))
    )
    if not raw_episodes:
        print(f"[convert] No raw episodes found under {args.raw_dir} — nothing to do.")
        return

    data_dir = os.path.join(args.out_dir, "data", "chunk-000")
    # Video subdirs MUST be named after the exact feature key (matching info.json's own
    # video_path template "videos/chunk-{episode_chunk:03d}/{video_key}/episode_....mp4",
    # where video_key = the literal features dict key below, e.g.
    # "observation.images.image") — confirmed as a real bug live: naming these "image" /
    # "wrist_image" made GR00T's own lerobot_episode_loader.py fail with "Could not open
    # input file" on every episode, since it looks up the folder by the full feature key,
    # not a shortened alias.
    video_image_dir = os.path.join(args.out_dir, "videos", "chunk-000", "observation.images.image")
    video_wrist_dir = os.path.join(args.out_dir, "videos", "chunk-000", "observation.images.wrist_image")
    meta_dir = os.path.join(args.out_dir, "meta")
    for d in (data_dir, video_image_dir, video_wrist_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    tasks: dict[str, int] = {}
    episodes_meta = []
    total_frames = 0
    ep_out_idx = 0
    fps_values = []  # per-episode measured fps, averaged into the dataset-wide info.json fps

    for ep_name in raw_episodes:
        ep_dir = os.path.join(args.raw_dir, ep_name)
        meta_path = os.path.join(ep_dir, "meta.json")
        frames_path = os.path.join(ep_dir, "frames.jsonl")
        scene_path = os.path.join(ep_dir, "scene.mp4")
        wrist_path = os.path.join(ep_dir, "wrist.mp4")
        if not all(os.path.exists(p) for p in (meta_path, frames_path, scene_path, wrist_path)):
            print(f"[convert] Skipping incomplete episode dir: {ep_name}")
            continue

        with open(meta_path) as f:
            ep_meta = json.load(f)
        instruction = ep_meta["instruction"]
        if instruction not in tasks:
            tasks[instruction] = len(tasks)
        task_index = tasks[instruction]

        states = []
        with open(frames_path) as f:
            for line in f:
                states.append(json.loads(line)["state"])
        if len(states) < 2:
            print(f"[convert] Skipping {ep_name} — too few frames ({len(states)})")
            continue
        actions = build_action_from_states(states)
        n_frames = len(states)
        ep_fps = float(ep_meta.get("fps", 20.0))

        out_scene = os.path.join(video_image_dir, f"episode_{ep_out_idx:06d}.mp4")
        out_wrist = os.path.join(video_wrist_dir, f"episode_{ep_out_idx:06d}.mp4")
        n_scene_frames = resize_video(scene_path, out_scene, ep_fps)
        n_wrist_frames = resize_video(wrist_path, out_wrist, ep_fps)
        if n_scene_frames != n_frames or n_wrist_frames != n_frames:
            print(f"[convert] WARNING {ep_name}: frame count mismatch "
                  f"(state={n_frames}, scene_video={n_scene_frames}, wrist_video={n_wrist_frames}) "
                  f"— truncating to the shortest.")
            n_frames = min(n_frames, n_scene_frames, n_wrist_frames)
            states, actions = states[:n_frames], actions[:n_frames]
        fps_values.append(ep_fps)

        table = {
            "observation.state": [np.asarray(s, dtype=np.float32) for s in states],
            "action": [np.asarray(a, dtype=np.float32) for a in actions],
            "timestamp": [round(i / ep_fps, 4) for i in range(n_frames)],
            "frame_index": list(range(n_frames)),
            "episode_index": [ep_out_idx] * n_frames,
            "index": [total_frames + i for i in range(n_frames)],
            "task_index": [task_index] * n_frames,
        }
        df = pd.DataFrame(table)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                        os.path.join(data_dir, f"episode_{ep_out_idx:06d}.parquet"))

        episodes_meta.append({"episode_index": ep_out_idx, "tasks": [instruction], "length": n_frames})
        total_frames += n_frames
        ep_out_idx += 1

    if ep_out_idx == 0:
        print("[convert] No usable episodes converted — aborting before writing meta files.")
        return

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task, idx in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # meta/modality.json — copied verbatim from the LIBERO demo dataset; Physyk's recorded
    # state/action layout was built to match this exactly (see groot_episode_recorder.py).
    src_modality = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Isaac-GR00T",
        "demo_data", "libero_demo", "meta", "modality.json",
    )
    shutil.copyfile(src_modality, os.path.join(meta_dir, "modality.json"))

    # Dataset-wide fps: LeRobot's info.json expects one value, but each episode's MEASURED
    # fps (from groot_episode_recorder.py) can vary slightly run to run — average them rather
    # than assume a fixed constant. Individual episodes were each re-encoded at their own
    # true fps above, so this average is only used for the dataset-level declared value.
    avg_fps = round(sum(fps_values) / len(fps_values), 2) if fps_values else 20.0
    if fps_values and (max(fps_values) - min(fps_values)) / avg_fps > 0.15:
        print(f"[convert] NOTE: per-episode fps varied more than 15% ({min(fps_values):.2f}-"
              f"{max(fps_values):.2f}) — dataset-wide info.json fps={avg_fps} is an average, "
              f"individual episode videos are each correctly encoded at their own true fps.")

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": ep_out_idx,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": ep_out_idx * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": avg_fps,
        "splits": {"train": f"0:{ep_out_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.wrist_image": {
                "dtype": "video", "shape": [256, 256, 3],
                "names": ["height", "width", "rgb"],
                "info": {"video.height": 256, "video.width": 256, "video.codec": "mp4v",
                          "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                          "video.fps": avg_fps, "video.channels": 3, "has_audio": False},
            },
            "observation.images.image": {
                "dtype": "video", "shape": [256, 256, 3],
                "names": ["height", "width", "rgb"],
                "info": {"video.height": 256, "video.width": 256, "video.codec": "mp4v",
                          "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                          "video.fps": avg_fps, "video.channels": 3, "has_audio": False},
            },
            "observation.state": {"dtype": "float32", "shape": [8],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]}},
            "action": {"dtype": "float32", "shape": [7],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"[convert] Wrote {ep_out_idx} episodes, {total_frames} frames, "
          f"{len(tasks)} distinct task strings to {args.out_dir}")
    print(
        "[convert] NOTE: video.codec is 'mp4v' here (cv2's baseline writer), not the 'av1' "
        "the checkpoint's own demo data used. GR00T's loader decodes video via torchcodec, "
        "which requires system FFmpeg shared libraries — NOT currently installed on this "
        "host (torchcodec import fails with 'Could not load libtorchcodec'). Install FFmpeg "
        "(matching torchcodec's supported versions 4-7) before running launch_finetune.py, "
        "or re-encode these videos to av1/yuv420p once FFmpeg is available, whichever is "
        "easier at that point."
    )


if __name__ == "__main__":
    main()
