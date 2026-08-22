#!/usr/bin/env python3
"""
NVIDIA Isaac-GR00T N1.7 VLA Policy Service — REAL inference (Phase 3)
======================================================================
Wraps the real `Gr00tPolicy` (Cosmos-Reason2-2B VLM backbone + DiT action head),
loaded from the LIBERO-finetuned checkpoint (models/GR00T-N1.7-LIBERO/libero_10 —
NOT the base GR00T-N1.7-3B pretrain checkpoint, which has no libero_sim embodiment
registered; see PLAN.md Phase 0).

This used to be a stub: `self.model = None`, with predictions produced by
`if/elif` keyword-matching on the instruction string. That's gone — this file now
loads real weights and runs a real forward pass every call.

Runs as a standalone FastAPI HTTP server (own process, own venv —
Isaac-GR00T/.venv/bin/python3 — because the pinned torch/transformers/flash-attn
build this model needs isn't installed in Isaac Sim's own bundled Python that
isaac_sim_service.py runs under). isaac_sim_service.py's VLA control loop is an
HTTP client of this server (see the `/predict` route), same pattern as
nemotron_fastapi_server.py.
"""

import io
import os
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure Isaac-GR00T package is on python path
GR00T_ROOT = Path(__file__).resolve().parent / "Isaac-GR00T"
if str(GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(GR00T_ROOT))

logging.basicConfig(level=logging.INFO, format="[GR00T-N1.7] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.environ.get(
    "GROOT_MODEL_PATH", "/opt/dlami/nvme/physyk/models/GR00T-N1.7-LIBERO/libero_10"
)
IMG_SIZE = 256  # LIBERO's native camera resolution; images are resized to this


class GR00TN17PolicyService:
    """Real inference wrapper around Isaac-GR00T's Gr00tPolicy for LIBERO_PANDA."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda:0",
        embodiment_tag: str = "LIBERO_PANDA",
    ):
        self.model_path = model_path
        self.device = device
        self.embodiment_tag = embodiment_tag
        self.policy = None
        self.video_keys = None
        self.state_keys = None
        self.action_keys = None
        self.language_key = None
        self.action_horizon = None

        log.info(f"Loading REAL Gr00tPolicy from {model_path} (embodiment={embodiment_tag}, device={device})...")
        t0 = time.time()
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        modality_config = self.policy.get_modality_config()
        self.video_keys = modality_config["video"].modality_keys
        self.state_keys = modality_config["state"].modality_keys
        self.action_keys = modality_config["action"].modality_keys
        self.language_key = modality_config["language"].modality_keys[0]
        self.action_horizon = len(modality_config["action"].delta_indices)
        log.info(
            f"Loaded in {time.time() - t0:.1f}s — video={self.video_keys} state={self.state_keys} "
            f"action={self.action_keys} horizon={self.action_horizon} language_key='{self.language_key}'"
        )

    def format_observation(
        self,
        scene_image: np.ndarray,
        wrist_image: np.ndarray,
        ee_pose: np.ndarray,
        gripper_width: float,
        instruction: str,
    ) -> dict[str, Any]:
        """Builds the real nested observation schema Gr00tPolicy.check_observation expects:
        video[key]: (1,1,H,W,3) uint8; state[key]: (1,1,D) float32 per field; language[key]: [[str]].

        Args:
            scene_image: (H, W, 3) uint8 RGB array from the overhead/scene camera
            wrist_image: (H, W, 3) uint8 RGB array from the wrist camera
            ee_pose: (6,) float array [x, y, z, roll, pitch, yaw]
            gripper_width: float, current gripper opening in meters
            instruction: natural language task string
        """
        def _prep_video(img: np.ndarray) -> np.ndarray:
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = np.array(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE)))
            return img[np.newaxis, np.newaxis, :, :, :3].astype(np.uint8)

        video_frames = {"image": scene_image, "wrist_image": wrist_image}
        video = {key: _prep_video(video_frames[key]) for key in self.video_keys if key in video_frames}

        field_values = {
            "x": ee_pose[0], "y": ee_pose[1], "z": ee_pose[2],
            "roll": ee_pose[3], "pitch": ee_pose[4], "yaw": ee_pose[5],
        }
        state = {}
        for key in self.state_keys:
            if key == "gripper":
                state[key] = np.array([[[gripper_width / 2.0, gripper_width / 2.0]]], dtype=np.float32)
            else:
                state[key] = np.array([[[field_values.get(key, 0.0)]]], dtype=np.float32)

        language = {self.language_key: [[instruction]]}
        return {"video": video, "state": state, "language": language}

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """Runs a REAL forward pass through the loaded DiT action head.

        Returns:
            action_chunk: (horizon, len(action_keys)) float32 array — real, unnormalized
            (physical-units) per-field deltas decoded by the model's own processor, in the
            field order of self.action_keys (LIBERO: x,y,z,roll,pitch,yaw,gripper).
        """
        instruction = next(iter(observation.get("language", {}).values()), [[""]])[0][0]
        log.info(f"Running real GR00T forward pass for task: '{instruction}'")
        t0 = time.time()
        action_dict, _info = self.policy.get_action(observation)
        dt = time.time() - t0
        chunk = np.stack([action_dict[key][0, :, 0] for key in self.action_keys], axis=-1)
        log.info(f"Inference took {dt:.2f}s -> action_chunk shape {chunk.shape}")
        return chunk.astype(np.float32)


# ─── FastAPI HTTP Server ────────────────────────────────────────────────────────
app = FastAPI(title="Physyk AI — GR00T-N1.7 LIBERO Policy Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

service: Optional[GR00TN17PolicyService] = None


def _decode_b64_image(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


class PredictRequest(BaseModel):
    instruction: str
    scene_image_b64: str
    wrist_image_b64: str
    ee_pose: list[float]  # [x, y, z, roll, pitch, yaw]
    gripper_width: float = 0.04


@app.on_event("startup")
def startup_event():
    global service
    service = GR00TN17PolicyService()


@app.get("/health")
def health():
    return {
        "status": "ok" if service is not None else "loading",
        "model_loaded": service is not None,
        "model_path": service.model_path if service else None,
        "action_horizon": service.action_horizon if service else None,
        "action_keys": service.action_keys if service else None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    try:
        scene_img = _decode_b64_image(req.scene_image_b64)
        wrist_img = _decode_b64_image(req.wrist_image_b64)
        obs = service.format_observation(
            scene_image=scene_img,
            wrist_image=wrist_img,
            ee_pose=np.array(req.ee_pose, dtype=np.float32),
            gripper_width=req.gripper_width,
            instruction=req.instruction,
        )
        chunk = service.predict_action_chunk(obs)
        return {
            "action_chunk": chunk.tolist(),
            "horizon": chunk.shape[0],
            "action_keys": service.action_keys,
        }
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8300)
