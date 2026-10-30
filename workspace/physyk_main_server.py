#!/usr/bin/env python3
"""
Physyk — Franka Panda Physical AI Control Server & Isaac Sim 6.0.1 Bridge
========================================================================
- Direct integration with real Isaac Sim 6.0.1 physics engine on port 8100
- Live Scene Camera & Wrist Camera MJPEG video streaming
- Real-time Franka Panda 7-DOF Joint & Gripper Telemetry
- Natural Language Instruction / Prompt Execution
- Served on port 7860
"""

import sys
import os
import re
import json
import subprocess
import time
import asyncio
import threading
import math
import base64
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Real agentic orchestrator (Nemotron decomposition + guardrails + dispatch/verify loop) — a pure HTTP
# client of the Nemotron (8000) and Isaac Sim (8100) servers.
# Non-agentic test mode (run.sh --no-agent, or PHYSYK_AGENTIC=0 directly) — skips loading
# the orchestrator entirely, so every instruction falls through the existing
# `if not AI_STACK_AVAILABLE` branch in execute_instruction() below and goes straight to
# Isaac Sim exactly as typed, no Nemotron decompose/guardrail/verify/retry loop. For
# comparing the raw PickPlaceController path against the agentic one.
AGENTIC_MODE = os.environ.get("PHYSYK_AGENTIC", "1") != "0"

if AGENTIC_MODE:
    try:
        from physyk_agent_orchestrator import NeMoRoboticsAgent
        nemo_agent = NeMoRoboticsAgent()
        AI_STACK_AVAILABLE = True
    except ImportError as e:
        AI_STACK_AVAILABLE = False
        print(f"Warning: Agentic orchestrator not loaded: {e}")
else:
    AI_STACK_AVAILABLE = False
    print("[Physyk] Non-agentic mode (PHYSYK_AGENTIC=0) — instructions dispatch directly "
          "to Isaac Sim, no Nemotron orchestrator.")

import physyk_config as cfg

ISAAC_SIM_URL = cfg.ISAAC_SIM_URL

app = FastAPI(title="Physyk Franka Panda AI Control")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Nemotron/Cosmos full-reasoning pages — /reasoning/nemotron, /reasoning/cosmos. Mounted here
# (same app, same port as everything else in this file) rather than on their own ports, since
# only 7860/8211 are actually forwarded through Brev/VS Code (see BREV_PORT_CONFIG.md) — a
# separate port would work over plain localhost but 404/fail to connect for anyone reaching
# this dashboard remotely, which is exactly what happened with the first version of this.
from reasoning_dashboard import mount_reasoning_routes, custom_pages
mount_reasoning_routes(app)


def _render_perception_log_page() -> str:
    """/reasoning/perception — what the overview camera detected, with the annotated frame
    (bounding boxes + estimated XYZ) for every pre-pick detection and post-place check. The
    data lives in the Isaac Sim process; images are proxied via /perception_log/<id>.jpg."""
    import html as _html
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/perception/log", headers={"User-Agent": "Physyk-Bridge"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"<h1 style='font-family:monospace;color:#f87171'>Isaac Sim unreachable: {_html.escape(str(e))}</h1>"
    rows = []
    for e in entries:
        lines = []
        for key, o in (e.get("objects") or {}).items():
            name = key.replace("_", " ").title()
            if o.get("seen"):
                x, y, z = o["camera_xyz_m"]
                lines.append(f"{name}: seen at ({x:.3f}, {y:.3f}, {z:.3f}) m"
                             + (f", yaw {o['yaw_deg']}°" if o.get("yaw_deg") is not None else "")
                             + (f" — scoring error vs sim {o['score_error_mm']} mm" if o.get("score_error_mm") is not None else ""))
            else:
                lines.append(f"{name}: NOT SEEN")
        rows.append(f"""<div class="entry">
          <div class="meta"><span class="t">{e['t']}</span> <span class="label">{_html.escape(e['event'])}</span></div>
          <div class="text">{_html.escape(e.get('note',''))}\n{_html.escape(chr(10).join(lines))}</div>
          <img class="frame" src="/perception_log/{e['id']}.jpg">
        </div>""")
    body = "".join(rows) or '<div class="empty">No perception events yet — switch on the perception toggle and run a pick-and-place.</div>'
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Perception — Log</title><style>
 body {{ background:radial-gradient(1000px 600px at 8% -10%, rgba(110,140,205,0.12), transparent 62%), #0a0c10; color:#e5e8ef;
        font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif; margin:0; padding:28px 32px; min-height:100vh; }}
 h1 {{ color:#eef1f7; font-size:1.05rem; font-weight:600; margin:0 0 18px; letter-spacing:0.2px; padding-bottom:12px; border-bottom:1px solid #232a38; }}
 .entry {{ border-left:2px solid #8ea3cf; background:rgba(22,27,36,0.7); border-top:1px solid #1c222e; border-right:1px solid #1c222e;
          border-bottom:1px solid #1c222e; padding:12px 16px; margin-bottom:12px; border-radius:0 8px 8px 0; }}
 .meta {{ font-size:0.72rem; color:#7e879a; margin-bottom:5px; }} .label {{ font-weight:600; color:#8ea3cf; }}
 .text {{ font-family:'JetBrains Mono',ui-monospace,monospace; font-size:0.8rem; line-height:1.55; white-space:pre-wrap; color:#d0d5e0; }}
 .empty {{ color:#59626f; }}
 .frame {{ display:block; margin-top:10px; max-width:100%; width:720px; border-radius:6px; border:1px solid #232a38; }}
</style></head><body><h1>Perception — camera detections (overview camera, color segmentation + depth)</h1>{body}</body></html>"""


custom_pages["perception"] = _render_perception_log_page

connected_ws: list = []
broadcast_lock = threading.Lock()

# ─── Fallback / Cached Simulation State ───────────────────────────────────────
sim_state = {
    "joints": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "ee_pos": [0.45, 0.0, 0.40],
    "gripper": 0.04,
    "stage": "READY",
    "busy": False,
    "last_instruction": "System Online",
    "fps": 60.0,
    "isaac_connected": False,
    "log": [],
    # Structured agent plan state (Phase 5) — driven by NeMoRoboticsAgent.run()'s
    # on_plan_update callback, not derived from the text log. Lets the UI show real
    # decompose/guardrail/retry/replan transitions as they happen, not just prose lines.
    "plan": None,
    "agentic_mode": AGENTIC_MODE,
    # Cosmos-Reason2 shadow-mode perception snapshot (cosmos_integration.md Step 6) — the
    # last grounding call's stage-vs-vision delta per object. Purely informational: nothing
    # in sim_state or the dispatch path reads this to make a decision.
    "perception": None,
    # Live online/offline status for the 3 backing services, independent of whether the
    # "Send to Agent" toggle is on — the toggle only changes dispatch ROUTING (see
    # /agent_toggle's own docstring), it never starts or stops anything, so this panel is
    # meant to reflect real process health regardless of that toggle's position.
    "service_status": {
        "isaac_sim": False,
        "nemotron": False,
        "cosmos": False,
    },
}

def push_log(msg, level="INFO"):
    entry = {"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    sim_state["log"].append(entry)
    if len(sim_state["log"]) > 200:
        sim_state["log"] = sim_state["log"][-100:]

async def broadcast(data: dict):
    dead = []
    for ws in list(connected_ws):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try: connected_ws.remove(ws)
        except: pass

def sync_broadcast(data: dict):
    try:
        loop = asyncio.get_event_loop()
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast(data), loop)
    except RuntimeError:
        pass

# ─── Isaac Sim Bridge Communication ───────────────────────────────────────────
def poll_isaac_sim_state():
    """Continuously syncs live telemetry from Isaac Sim on port 8100."""
    _perception_tick = 0
    while True:
        try:
            req = urllib.request.Request(f"{ISAAC_SIM_URL}/state", headers={"User-Agent": "Physyk-Bridge"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    sim_state["joints"] = data.get("joints", sim_state["joints"])
                    sim_state["gripper"] = data.get("gripper", sim_state["gripper"])
                    sim_state["ee_pos"] = data.get("ee_pos", sim_state["ee_pos"])
                    sim_state["stage"] = data.get("stage", "READY")
                    sim_state["busy"] = data.get("busy", False)
                    sim_state["last_instruction"] = data.get("last_instruction", sim_state["last_instruction"])
                    sim_state["fps"] = data.get("fps", 60.0)
                    sim_state["isaac_connected"] = True
        except Exception:
            sim_state["isaac_connected"] = False

        # Perception snapshot only changes once per dispatched pick/place instruction (not
        # every frame), so polling it at the full 10 Hz telemetry rate would be pure waste —
        # 1 Hz is plenty and matches the cadence Isaac Sim itself updates depth stats at.
        _perception_tick += 1
        if _perception_tick >= 10:
            _perception_tick = 0
            try:
                preq = urllib.request.Request(f"{ISAAC_SIM_URL}/perception", headers={"User-Agent": "Physyk-Bridge"})
                with urllib.request.urlopen(preq, timeout=1.0) as presp:
                    if presp.status == 200:
                        sim_state["perception"] = json.loads(presp.read().decode("utf-8"))
            except Exception:
                pass  # keep the last-known snapshot rather than blanking it on a transient miss

        time.sleep(0.1)  # 10 Hz telemetry sync

state_sync_thread = threading.Thread(target=poll_isaac_sim_state, daemon=True)
state_sync_thread.start()

def _url_is_healthy(url: str, timeout: float = 5.0, api_key: str = "") -> bool:
    try:
        headers = {"User-Agent": "Physyk-Bridge"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False

def poll_service_status():
    """Independent, low-frequency health poll for the 3 backing services NOT already
    covered by poll_isaac_sim_state's 10Hz /state loop (isaac_sim's own status is copied
    from sim_state["isaac_connected"], not re-polled here, to avoid a second concurrent
    check hitting the same endpoint). Deliberately slow (5s) — these are startup-status
    indicators for a dashboard, not something anything else in the app reads to make a
    decision, so there's no reason to poll them at telemetry speed."""
    while True:
        # Hosted NVIDIA API (or a local OpenAI-compatible server): "online" = the /models
        # listing answers with our key. Slow poll — every call is a real network round trip.
        sim_state["service_status"]["nemotron"] = _url_is_healthy(f"{cfg.NEMOTRON_BASE_URL}/models", api_key=cfg.NEMOTRON_API_KEY)
        sim_state["service_status"]["cosmos"] = _url_is_healthy(f"{cfg.COSMOS_BASE_URL}/models", api_key=cfg.COSMOS_API_KEY)
        time.sleep(30.0)

SIM_CONTAINER_NAME = os.environ.get("PHYSYK_SIM_CONTAINER", "physyk-isaac-sim")

def _sim_container_running() -> bool:
    """True if the Isaac Sim docker container (./run.sh --sim) is running — it counts as ONLINE even while the
    sim inside is still loading and port 8100 does not answer yet."""
    try:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", SIM_CONTAINER_NAME],
                             capture_output=True, text=True, timeout=3)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except Exception:
        return False

def poll_isaac_status():
    """Isaac Sim pill: ONLINE if its docker container is running OR its API answers (a windowed/native run)."""
    while True:
        sim_state["service_status"]["isaac_sim"] = bool(sim_state["isaac_connected"]) or _sim_container_running()
        time.sleep(3.0)

threading.Thread(target=poll_isaac_status, daemon=True).start()

service_status_thread = threading.Thread(target=poll_service_status, daemon=True)
service_status_thread.start()

def _forward_direct_to_isaac(instruction: str):
    """Fire-and-forget single instruction straight to Isaac Sim's /execute — the original
    dispatch path, kept as-is for plain control commands (reset/home/randomize) that
    don't need LLM reasoning and that Isaac Sim's own loop already keyword-detects reliably."""
    try:
        req = urllib.request.Request(
            f"{ISAAC_SIM_URL}/execute",
            data=json.dumps({"instruction": instruction}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            json.loads(resp.read().decode("utf-8"))
            push_log(f"Dispatched to Isaac Sim (PhysX + RMPFlow): '{instruction}'", "SUCCESS")
            return {"status": "started", "instruction": instruction, "isaac_sim": True}
    except Exception as e:
        push_log(f"Could not reach Isaac Sim: {e}", "WARN")
        return {"status": "started", "instruction": instruction, "isaac_sim": False}

# Live, in-browser toggle (the "Send to Agent" switch) — distinct from AGENTIC_MODE above,
# which is a startup-time capability check (did the orchestrator even load). This is a
# runtime preference: with the orchestrator loaded and available, the user can still flip
# every prompt to bypass it and go straight to Isaac Sim, without restarting anything.
# Starts wherever AGENTIC_MODE left it (off entirely if the orchestrator never loaded).
agent_enabled = AGENTIC_MODE

# Only one orchestrated multi-step plan may run at a time — dispatching a second one while
# the first is still stepping through subgoals would race both against Isaac Sim's own
# single-command busy lock. Plain control commands below bypass this entirely.
_orchestrator_lock = threading.Lock()

# Generation counter guarding the lock release below against a stale-thread race: /hard_reset
# force-releases _orchestrator_lock so a new prompt isn't stuck waiting on a plan that's being
# aborted (the aborting plan's own background thread can lag behind by up to one HTTP
# poll/timeout before it actually notices the stop and unwinds). Without this guard, that late
# `finally` release could fire *after* a brand-new plan has already re-acquired the same lock —
# silently releasing a lock it doesn't own and letting two orchestrated plans run concurrently
# against Isaac Sim's single command queue. Each orchestrated run captures the generation at
# start and only releases if it's still current — a forced release in between bumps the
# generation, so a late/stale run's release becomes a harmless no-op instead of stealing the
# next run's lock.
_orchestrator_lock_gen = 0
_orchestrator_lock_gen_lock = threading.Lock()

_CONTROL_KEYWORDS = ("reset", "home", "park", "retract", "randomize", "shuffle")

def _is_control_command(instruction: str) -> bool:
    inst = instruction.strip().lower()
    return any(inst == kw or inst.startswith(kw + " ") or inst == f"go {kw}" for kw in _CONTROL_KEYWORDS)

def _on_plan_update(plan: dict):
    sim_state["plan"] = plan

def _run_orchestrated_plan(instruction: str, my_gen: int):
    try:
        nemo_agent.run(instruction, push_log=push_log, on_plan_update=_on_plan_update)
    except Exception as e:
        push_log(f"Orchestrator error: {e}", "ERROR")
    finally:
        with _orchestrator_lock_gen_lock:
            still_current = (_orchestrator_lock_gen == my_gen)
        if still_current and _orchestrator_lock.locked():
            try:
                _orchestrator_lock.release()
            except RuntimeError:
                pass

def execute_instruction(instruction: str):
    """Real Phase 5 dispatch: plain control commands go straight to Isaac Sim as before;
    everything else is decomposed/guardrailed/dispatched by the real Nemotron-backed
    orchestrator (physyk_agent_orchestrator.NeMoRoboticsAgent), run in a background thread so
    a multi-step plan's blocking HTTP polling never stalls this server's shared event loop
    (camera streaming, /state polling for the GUI). Falls back to the direct single-shot path
    if the orchestrator isn't available, or if the "Send to Agent" switch is off."""
    push_log(f"Command received: '{instruction}'", "INFO")

    if not AI_STACK_AVAILABLE or not agent_enabled or _is_control_command(instruction):
        return _forward_direct_to_isaac(instruction)

    if not _orchestrator_lock.acquire(blocking=False):
        push_log("Orchestrator is already running a plan — ignoring new command until it finishes.", "WARN")
        return {"status": "busy", "instruction": instruction}

    with _orchestrator_lock_gen_lock:
        global _orchestrator_lock_gen
        _orchestrator_lock_gen += 1
        my_gen = _orchestrator_lock_gen

    threading.Thread(target=_run_orchestrated_plan, args=(instruction, my_gen), daemon=True).start()
    return {"status": "started", "instruction": instruction, "orchestrated": True}

# ─── Camera Streaming Proxies (Zero-Lag Async httpx) ───────────────────────────
import httpx

async def proxy_mjpeg_stream(stream_path: str):
    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", f"{ISAAC_SIM_URL}{stream_path}") as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                img[:] = (46, 11, 5)                                  # deep navy (BGR), same as the page background
                cv2.putText(img, "CONNECTING TO ISAAC SIM...", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (247, 230, 223), 1, cv2.LINE_AA)
                _, encoded = cv2.imencode(".jpg", img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/stream")
async def proxy_scene_stream():
    return await proxy_mjpeg_stream("/camera/stream")

@app.get("/camera/scene.stream")
async def proxy_scene_stream_alias():
    return await proxy_mjpeg_stream("/camera/stream")

@app.get("/camera/wrist.stream")
async def proxy_wrist_stream():
    return await proxy_mjpeg_stream("/camera/wrist.stream")

@app.get("/camera/{cam_name}.jpg")
def proxy_camera_jpeg(cam_name: str):
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/camera/{cam_name}.jpg")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            return Response(content=response.read(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)

@app.get("/api/cameras")
def get_cameras_api():
    return {
        "cameras": [
            {"id": "scene", "name": "3rd Person Isometric Overview", "stream": "/camera/stream"},
            {"id": "wrist", "name": "Wrist Camera (Gripper Close-up)", "stream": "/camera/wrist.stream"},
        ]
    }

@app.get("/api/lighting")
def get_lighting_api():
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/api/lighting")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return JSONResponse(json.loads(resp.read().decode("utf-8")))
    except Exception:
        return {
            "current": "studio",
            "presets": [
                {"id": "studio", "name": "Studio White (Default)"},
                {"id": "high_contrast", "name": "High-Contrast Inspection"},
                {"id": "warm", "name": "Warm Industrial"},
                {"id": "cyberpunk", "name": "Cyberpunk Neon"},
                {"id": "neutral", "name": "Neutral Daylight"}
            ]
        }

@app.post("/lighting")
async def proxy_lighting(req: Request):
    try:
        body = await req.body()
        preset = json.loads(body.decode("utf-8")).get("preset", "?") if body else "?"
        forward_req = urllib.request.Request(
            f"{ISAAC_SIM_URL}/lighting",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(forward_req, timeout=2.0) as resp:
            push_log(f"Lighting preset changed: '{preset}'", "SUCCESS")
            return JSONResponse(json.loads(resp.read().decode("utf-8")))
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/reset")
def proxy_reset():
    push_log("Reset requested — homing arm, returning cubes to staging spots...", "INFO")
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/reset", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return JSONResponse(json.loads(resp.read().decode("utf-8")))
    except Exception as e:
        push_log(f"Reset request failed: {e}", "WARN")
        return {"status": "resetting"}

@app.post("/hard_reset")
def hard_reset():
    """Emergency stop for the GUI's red 'Hard Reset' button.
    Immediately aborts whatever's running (an orchestrated multi-step plan and/or the sim's
    own in-flight task), clears it so the next prompt isn't blocked by the busy lock, homes
    the arm, AND returns every cube to its default staging spot. Two independent things need
    stopping, so both are signalled here:
      1. The orchestrator's own run() loop (physyk_agent_orchestrator.NeMoRoboticsAgent),
         if a multi-step plan is mid-flight — via request_stop(), checked at every
         poll/loop boundary so it bails out within one poll interval instead of running a
         subgoal, a retry, or a Nemotron replan call to completion first.
      2. Isaac Sim's own physics loop, which may be mid-motion on a pick_place task
         dispatched either by the orchestrator or directly — via isaac_sim_service.py's own
         /hard_stop, which drops whatever's active, homes the arm, and resets every cube."""
    push_log("HARD RESET — stopping any active plan/task, homing the arm, and resetting cubes...", "WARN")

    if AI_STACK_AVAILABLE:
        try:
            nemo_agent.request_stop()
        except Exception as e:
            push_log(f"Could not signal orchestrator to stop: {e}", "WARN")

    isaac_ok = False
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/hard_stop", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            json.loads(resp.read().decode("utf-8"))
            isaac_ok = True
    except Exception as e:
        push_log(f"Could not reach Isaac Sim for hard stop: {e}", "WARN")

    # Forcibly free the orchestrator lock so a new prompt is accepted immediately instead of
    # bouncing off "busy" until the aborting run's own background thread unwinds (which can
    # lag behind — e.g. it may be blocked inside an in-flight Nemotron HTTP call for up to
    # NEMOTRON_TIMEOUT_S before it next checks the stop flag). Bump the generation counter
    # first so that thread's own later `finally` release becomes a no-op instead of releasing
    # whatever new plan has since re-acquired this same lock — see _run_orchestrated_plan.
    with _orchestrator_lock_gen_lock:
        global _orchestrator_lock_gen
        _orchestrator_lock_gen += 1
    if _orchestrator_lock.locked():
        try:
            _orchestrator_lock.release()
        except RuntimeError:
            pass

    sim_state["plan"] = None
    sim_state["last_instruction"] = "Hard reset — ready for new commands"
    push_log("Hard reset complete — ready for a new instruction.", "SUCCESS")
    return {"status": "stopped", "isaac_sim": isaac_ok}

@app.post("/randomize")
def proxy_randomize():
    push_log("Randomizing cube positions (genuinely random, arm-reachable, tray-clear)...", "INFO")
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/randomize", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return JSONResponse(json.loads(resp.read().decode("utf-8")))
    except Exception as e:
        push_log(f"Randomize request failed: {e}", "WARN")
        return {"status": "error", "message": str(e)}

@app.get("/perception_log/{entry_id}.jpg")
def perception_log_image(entry_id: int):
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/perception/log/{entry_id}.jpg", headers={"User-Agent": "Physyk-Bridge"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return Response(content=resp.read(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)

@app.post("/perception_mode")
async def perception_mode(req: Request):
    """The "Drive arm from camera" switch: flips Isaac Sim's perception provider between
    'shadow' (deterministic — simulator ground-truth poses drive the arm) and 'vision'
    (pure perception: camera-derived poses only, no simulator coordinates, no fallback). Live,
    no restart; takes effect on the next dispatched pick/place."""
    body = await req.json()
    mode = "pure" if body.get("vision") else "shadow"
    try:
        preq = urllib.request.Request(
            f"{ISAAC_SIM_URL}/perception/mode",
            data=json.dumps({"mode": mode}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(preq, timeout=5.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        push_log(f"Perception switch: arm is now driven by "
                 f"{'CAMERA ONLY (pure perception — no simulator coordinates)' if mode == 'pure' else 'simulator ground truth (deterministic)'}.", "INFO")
        return {"status": "ok", "mode": result.get("mode", mode)}
    except Exception as e:
        push_log(f"Could not switch perception mode: {e}", "WARN")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)

@app.post("/agent_toggle")
async def agent_toggle(req: Request):
    """The "Send to Agent" switch — a live, in-browser toggle distinct from the AGENTIC_MODE
    startup flag (run.sh --no-agent): with the orchestrator loaded and available, this lets
    the user flip every subsequent prompt between the full Nemotron decompose/guardrail/
    verify/retry loop and a direct pass-through to Isaac Sim, without restarting anything.
    While off, the Agent Plan panel goes on hold (sim_state["plan"] cleared) rather than
    showing stale plan data from before the switch was flipped."""
    global agent_enabled
    body = await req.json()
    agent_enabled = bool(body.get("enabled", True)) and AI_STACK_AVAILABLE
    sim_state["agentic_mode"] = agent_enabled
    if not agent_enabled:
        sim_state["plan"] = None
    push_log(f"Send-to-Agent switch turned {'ON' if agent_enabled else 'OFF'} — prompts now "
             f"go {'through the Nemotron orchestrator' if agent_enabled else 'directly to Isaac Sim'}.",
             "INFO")
    return {"status": "ok", "agentic_mode": agent_enabled}

# ─── HTML USER INTERFACE ──────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Physyk AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c10;
    --bg-2: #0d1016;
    --surface: #11141a;
    --card: #161b24;
    --card-2: #1a2029;
    --border: #232a38;
    --border-soft: #1c222e;
    --accent: #8ea3cf;
    --accent-glow: rgba(142, 163, 207, 0.10);
    --accent2: #a9b8d6;
    --cyan: #86a5c8;
    --green: #7fb4c9;
    --amber: #b3a98f;
    --red: #b88a93;
    --ok: #86b39d;          /* dull green: running / done / success */
    --bad: #c4807f;         /* dull red: offline / failed */
    --coral: #e0705f;       /* hard reset */
    --text: #eef1f7;
    --muted: #7e879a;
    --muted-2: #59626f;
    --r: 12px;
    --r-sm: 9px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scrollbar-color: #2a3142 transparent; }
  body {
    background:
      radial-gradient(1200px 700px at 12% -10%, rgba(142,163,207,0.08), transparent 60%),
      radial-gradient(900px 600px at 100% 0%, rgba(169,184,214,0.06), transparent 55%),
      var(--bg);
    color: var(--text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh; overflow-x: hidden;
  }
  ::-webkit-scrollbar { width: 9px; height: 9px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2a3142; border-radius: 999px; }
  ::-webkit-scrollbar-thumb:hover { background: #333d50; }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
  .display { font-family: 'Sora', 'Inter', sans-serif; }
  .icon, .icon-sm, .icon-lg { flex-shrink: 0; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; vertical-align: -3px; }
  .icon { width: 15px; height: 15px; }
  .icon-sm { width: 12px; height: 12px; vertical-align: -2px; }
  .icon-lg { width: 22px; height: 22px; }

  /* ── HEADER ── */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 26px; border-bottom: 1px solid var(--border-soft);
    background: rgba(13,16,22,0.92); backdrop-filter: blur(14px) saturate(140%);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 13px; }
  .logo-icon {
    width: 38px; height: 38px; background: linear-gradient(135deg,#8ea3cf,#a9b8d6); border-radius: 11px;
    display: flex; align-items: center; justify-content: center; font-size: 19px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08) inset, 0 4px 16px var(--accent-glow);
  }
  .logo-text h1 { font-family: 'Sora', 'Inter', sans-serif; font-size: 1.08rem; font-weight: 700; letter-spacing: -0.3px; }
  .logo-text p  { font-size: 0.72rem; color: var(--muted); margin-top: 1px; }

  .header-badges { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .status-pill {
    display: flex; align-items: center; gap: 8px; padding: 6px 14px;
    background: var(--card); border: 1px solid var(--border); border-radius: 999px;
    font-size: 0.76rem; font-weight: 600;
  }
  .status-pill.feature { color: var(--cyan); border-color: rgba(134,165,200,0.28); background: rgba(134,165,200,0.08); }
  .status-pill.feature .pill-icon { filter: drop-shadow(0 0 4px rgba(134,165,200,0.6)); }
  .status-pill.mode-agentic { color: var(--accent2); border-color: rgba(169,184,214,0.28); background: rgba(169,184,214,0.08); }
  .status-pill.mode-direct { color: var(--amber); border-color: rgba(179,169,143,0.28); background: rgba(179,169,143,0.08); }
  .status-pill.mode-shadow, .status-pill.mode-stage { color: var(--cyan); border-color: rgba(134,165,200,0.28); background: rgba(134,165,200,0.08); }
  .status-pill.mode-vision, .status-pill.mode-pure { color: var(--green); border-color: rgba(127,180,201,0.28); background: rgba(127,180,201,0.08); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  .dot.busy { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 0.6s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* 4-service status strip — real process health for Isaac Sim / Nemotron / Cosmos,
     independent of the agent toggle (a service can be ONLINE while the toggle routes
     around it). See poll_service_status() in the Python backend. */
  .svc-strip { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .svc-pill {
    display: flex; align-items: center; gap: 6px; padding: 5px 11px;
    background: var(--card); border: 1px solid var(--border); border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
  }
  .svc-pill .svc-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .svc-pill.svc-online { color: var(--green); border-color: rgba(127,180,201,0.28); background: rgba(127,180,201,0.08); }
  .svc-pill.svc-online .svc-dot { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .svc-pill.svc-offline { color: var(--muted); border-color: var(--border); background: rgba(126,135,154,0.06); }
  .svc-pill.svc-offline .svc-dot { background: var(--muted); box-shadow: none; }
  .svc-pill .svc-state { opacity: 0.75; font-weight: 500; font-size: 0.68rem; }

  /* ── LAYOUT ── */
  .layout { display: grid; grid-template-columns: 1fr 410px; gap: 0; height: calc(100vh - 64px); }

  /* ── VIEWPORT ── */
  .viewport-panel { padding: 18px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }
  .viewport-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--r);
    overflow: hidden; flex: 1; position: relative; min-height: 440px; display: flex; flex-direction: column;
    box-shadow: 0 14px 40px rgba(0,0,0,0.45);
  }
  .viewport-header {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    padding: 11px 16px; border-bottom: 1px solid var(--border-soft);
    background: rgba(15,18,24,0.9); z-index: 10;
  }
  .viewport-title { font-size: 0.81rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .vp-select {
    background: var(--card-2); color: #a9b8d6; border: 1px solid var(--border); border-radius: 7px;
    padding: 5px 9px; font-size: 0.74rem; outline: none; cursor: pointer; font-weight: 600; font-family: inherit;
  }
  .vp-select:hover { border-color: #333d50; }
  .vp-badge { font-size: 0.68rem; padding: 3px 10px; border-radius: 999px; background: rgba(142,163,207,0.14); color: var(--accent); border: 1px solid rgba(142,163,207,0.28); font-weight: 700; }
  .fps-readout { font-size: 0.74rem; color: var(--muted); font-weight: 600; }
  .fps-readout .live-tag { color: var(--green); font-size: 0.62rem; font-weight: 800; letter-spacing: 0.4px; margin-right: 5px; }

  .stream-container {
    flex: 1; position: relative; background: #030507; display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  .stream-video { width: 100%; height: 100%; object-fit: contain; display: block; }

  .stage-overlay {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(9,11,15,0.82); backdrop-filter: blur(10px); border: 1px solid var(--border);
    border-radius: 999px; padding: 7px 20px; font-size: 0.78rem; font-weight: 700;
    color: var(--accent); letter-spacing: 0.4px; box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    display: flex; align-items: center; gap: 8px;
  }
  .stage-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
  .stage-overlay.idle { color: var(--green); }
  .stage-overlay.idle .stage-dot { background: var(--green); }

  /* ── TELEMETRY ROW ── */
  .telemetry-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; }
  .telem-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--r-sm);
    padding: 13px 14px; transition: border-color 0.2s, transform 0.2s;
  }
  .telem-card:hover { border-color: #333d50; transform: translateY(-1px); }
  .telem-label {
    font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.9px; font-weight: 700;
    margin-bottom: 7px; display: flex; align-items: center; justify-content: space-between; gap: 6px;
  }
  .telem-tag { font-size: 0.56rem; padding: 1px 6px; border-radius: 999px; font-weight: 800; letter-spacing: 0.3px; }
  .telem-tag.live { background: rgba(127,180,201,0.14); color: var(--green); }
  .telem-tag.cfg { background: rgba(126,135,154,0.16); color: var(--muted); }
  .telem-val { font-size: 1.08rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
  .telem-unit { font-size: 0.68rem; color: var(--muted); margin-left: 3px; font-weight: 500; }
  .telem-gauge-bg { height: 4px; background: var(--border); border-radius: 999px; overflow: hidden; margin-top: 8px; }
  .telem-gauge-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,var(--accent),var(--accent2)); transition: width 0.15s ease; }
  .telem-xyz { display: flex; gap: 10px; font-size: 0.92rem; }
  .telem-xyz span { color: var(--muted-2); font-weight: 600; font-size: 0.68rem; margin-right: 2px; }

  /* ── JOINT BARS ── */
  .joints-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 18px;
  }
  .joints-title {
    font-size: 0.70rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.9px; font-weight: 700;
    margin-bottom: 13px; display: flex; align-items: center; gap: 8px;
  }
  .joints-title::after { content:''; flex:1; height:1px; background: var(--border-soft); }
  .joint-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 11px 22px; }
  .joint-row { display: flex; align-items: center; gap: 10px; }
  .joint-name { font-size: 0.68rem; color: var(--muted); width: 24px; text-align: right; font-weight: 700; }
  .joint-bar-bg { flex: 1; height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
  .joint-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,var(--accent),var(--accent2)); transition: width 0.12s ease; }
  .joint-val { font-size: 0.70rem; color: var(--text); width: 48px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }

  /* ── SIDEBAR ── */
  .sidebar { background: var(--surface); border-left: 1px solid var(--border-soft); display: flex; flex-direction: column; overflow-y: auto; }

  /* ── PROMPT PANEL ── */
  .prompt-panel { padding: 20px; border-bottom: 1px solid var(--border-soft); }
  .section-title {
    font-size: 0.70rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.1px; font-weight: 700;
    margin-bottom: 13px; display: flex; align-items: center; gap: 8px;
  }
  .section-title::after { content:''; flex:1; height:1px; background: var(--border-soft); }

  .prompt-box {
    width: 100%; background: var(--card); border: 1px solid var(--border); border-radius: var(--r-sm);
    color: var(--text); font-size: 0.89rem; padding: 13px 15px; resize: none; outline: none;
    font-family: inherit; transition: all 0.2s; min-height: 80px; line-height: 1.5;
  }
  .prompt-box:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
  .prompt-box::placeholder { color: var(--muted-2); }

  .btn-row { display: flex; gap: 9px; margin-top: 11px; }

  .agent-switch-row { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
  .switch { position: relative; display: inline-block; width: 40px; height: 22px; flex-shrink: 0; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .switch-track {
    position: absolute; inset: 0; background: var(--card-2); border: 1px solid var(--border);
    border-radius: 999px; cursor: pointer; transition: background 0.2s ease, border-color 0.2s ease;
  }
  .switch-thumb {
    position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%;
    background: var(--muted); transition: transform 0.2s ease, background 0.2s ease;
  }
  .switch input:checked + .switch-track { background: rgba(169,184,214,0.22); border-color: var(--accent2); }
  .switch input:checked + .switch-track .switch-thumb { transform: translateX(18px); background: var(--accent2); }
  .agent-switch-label { font-size: 0.78rem; font-weight: 600; color: var(--muted); }
  .agent-switch-row.on .agent-switch-label { color: var(--text); }
  .btn {
    flex: 1; padding: 11px 16px; border: none; border-radius: 9px; cursor: pointer;
    font-size: 0.83rem; font-weight: 700; font-family: inherit; transition: all 0.15s ease;
  }
  .btn-primary { background: linear-gradient(135deg,#8ea3cf,#a9b8d6); color: #fff; box-shadow: 0 4px 14px var(--accent-glow); }
  .btn-primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 7px 22px rgba(142,163,207,0.38); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
  .btn-secondary { background: var(--card-2); border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { background: #232b39; border-color: #333d50; }
  .btn-danger { background: linear-gradient(135deg,#a97480,#7d4f58); color: #fff; box-shadow: 0 4px 14px rgba(169,116,128,0.35); }
  .btn-danger:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 7px 22px rgba(169,116,128,0.5); }
  .btn-danger:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* ── QUICK TASKS ── */
  .quick-cmds { padding: 18px 20px; border-bottom: 1px solid var(--border-soft); }
  .cmd-grid { display: grid; grid-template-columns: 1fr; gap: 7px; }
  .cmd-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 9px;
    padding: 10px 14px; font-size: 0.79rem; cursor: pointer; display: flex; align-items: center; gap: 10px;
    transition: all 0.15s ease; font-weight: 500;
  }
  .cmd-card:hover { border-color: var(--accent); background: rgba(142,163,207,0.07); transform: translateX(2px); }
  .cmd-swatch { width: 9px; height: 9px; border-radius: 3px; flex-shrink: 0; }

  /* ── API DIRECTORY ── */
  .api-row {
    display:flex; justify-content:space-between; align-items:center; padding:7px 11px;
    background:var(--card); border:1px solid var(--border); border-radius:8px; font-size: 0.77rem;
  }
  .api-row a { color: var(--accent); font-weight: 600; text-decoration: none; }
  .api-row a:hover { text-decoration: underline; }

  /* ── LOGS ── */
  .logs-panel { padding: 18px 20px; flex: 1; display: flex; flex-direction: column; min-height: 210px; }
  .logs-box {
    flex: 1; background: #07090d; border: 1px solid var(--border); border-radius: 9px;
    padding: 11px 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.71rem;
    overflow-y: auto; max-height: 260px; display: flex; flex-direction: column; gap: 5px;
  }
  .log-entry { display: flex; gap: 8px; line-height: 1.45; }
  .log-time { color: var(--muted-2); flex-shrink: 0; }
  .log-msg { color: #c6ccd8; word-break: break-word; }
  .log-msg.SUCCESS { color: #7fb4c9; }
  .log-msg.WARN { color: #b3a98f; }
  .log-msg.ERROR { color: #b88a93; }

  /* ── AGENT PLAN (Phase 5 — real Nemotron decompose/guardrail/retry/replan state) ── */
  .plan-panel { padding: 18px 20px; border-bottom: 1px solid var(--border-soft); }
  .plan-empty { font-size: 0.74rem; color: var(--muted); }
  .plan-instruction {
    font-size: 0.76rem; color: #c6ccd8; margin-bottom: 10px; line-height: 1.4;
    padding: 8px 10px; background: var(--card); border: 1px solid var(--border-soft); border-radius: 7px;
  }
  .plan-instruction .plan-status {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
    padding: 2px 8px; border-radius: 999px; margin-right: 8px;
  }
  .plan-steps { display: flex; flex-direction: column; gap: 6px; }
  .plan-step {
    display: flex; align-items: flex-start; gap: 9px; padding: 8px 10px;
    background: #07090d; border: 1px solid var(--border-soft); border-radius: 8px;
    font-size: 0.73rem; transition: border-color 0.2s ease;
  }
  .plan-step-idx {
    flex-shrink: 0; width: 18px; height: 18px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-size: 0.64rem; font-weight: 700;
    background: var(--card); color: var(--muted); margin-top: 1px;
  }
  .plan-step-body { flex: 1; min-width: 0; }
  .plan-step-text { color: #c6ccd8; word-break: break-word; }
  .plan-step-detail { color: var(--muted); font-size: 0.68rem; margin-top: 2px; }
  .plan-step-badge {
    flex-shrink: 0; display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; padding: 2px 7px; border-radius: 999px; white-space: nowrap;
  }
  .plan-step.PENDING { opacity: 0.55; }
  .badge-PENDING, .plan-status-PLANNING { background: rgba(126,135,154,0.16); color: var(--muted); }
  .plan-step.RUNNING { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent-glow); }
  .badge-RUNNING, .plan-status-PLANNING.active, .plan-status-RUNNING { background: rgba(142,163,207,0.16); color: var(--accent); }
  .plan-step.RETRYING, .plan-step.REPLANNING { border-color: #b3a98f; }
  .badge-RETRYING, .badge-REPLANNING, .plan-status-PARTIAL, .plan-status-VERIFYING { background: rgba(179,169,143,0.16); color: #b3a98f; }
  .plan-step.DONE { border-color: rgba(127,180,201,0.4); }
  .badge-DONE, .plan-status-SUCCESS { background: rgba(127,180,201,0.16); color: #7fb4c9; }
  .plan-step.FAILED, .plan-step.REJECTED { border-color: rgba(184,138,147,0.4); }
  .badge-FAILED, .badge-REJECTED, .plan-status-FAILED, .plan-status-REJECTED, .plan-status-GOAL_NOT_HELD { background: rgba(184,138,147,0.16); color: #b88a93; }
  .plan-step.ABORTED { opacity: 0.4; }
  .badge-ABORTED { background: rgba(126,135,154,0.16); color: var(--muted); }
  @keyframes plan-pulse { 0%,100% { box-shadow: 0 0 0 1px var(--accent-glow); } 50% { box-shadow: 0 0 0 4px var(--accent-glow); } }
  .plan-step.RUNNING, .plan-step.REPLANNING { animation: plan-pulse 1.6s ease-in-out infinite; }

  /* ── FOOTER STRIP ── */
  footer {
    padding: 8px 20px; border-top: 1px solid var(--border-soft); background: var(--surface);
    display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap;
  }
  .stack-chip {
    font-size: 0.64rem; color: var(--muted); font-weight: 600; padding: 3px 9px;
    background: var(--card); border: 1px solid var(--border-soft); border-radius: 999px;
  }

  /* ── Theme layer: the dark neutral base of the original UI, muted steel-blue accents, no glows ── */
  body {
    background: radial-gradient(1100px 650px at 10% -12%, rgba(110,140,205,0.07), transparent 60%), var(--bg);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  }
  header { background: rgba(13,16,22,0.94); border-bottom: 1px solid var(--border-soft); }
  .logo-text h1 { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1.05rem; letter-spacing: 0.2px; }
  .status-pill, .svc-pill { background: var(--card); border-color: var(--border); color: var(--text); font-weight: 500; }
  .status-pill.feature, .status-pill.mode-agentic, .status-pill.mode-direct, .status-pill.mode-shadow, .status-pill.mode-stage,
  .status-pill.mode-vision, .status-pill.mode-pure { color: var(--accent); border-color: var(--border); background: rgba(142,163,207,0.07); }
  .status-pill.feature .pill-icon { filter: none; }
  .svc-pill.svc-online { color: var(--text); border-color: rgba(127,180,201,0.30); background: rgba(127,180,201,0.07); }
  .svc-pill.svc-online .svc-dot, .dot, .dot.busy { box-shadow: none; }
  .svc-pill.svc-offline { color: var(--muted); }
  .dot { background: var(--green); }
  .dot.busy { background: var(--accent); }
  .btn { border-radius: 8px; font-weight: 600; letter-spacing: 0.2px; }
  .btn-primary { background: #3f5b8f; color: #fff; box-shadow: none; }
  .btn-primary:hover:not(:disabled) { background: #4a6aa3; transform: none; box-shadow: none; }
  .btn-secondary { background: var(--card-2); border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { background: #232b39; border-color: #333d50; }
  .btn-danger { background: transparent; color: #b88a93; border: 1px solid rgba(184,138,147,0.45); box-shadow: none; }
  .btn-danger:hover:not(:disabled) { background: rgba(184,138,147,0.10); transform: none; box-shadow: none; }
  .telem-gauge-fill, .joint-bar-fill { background: linear-gradient(90deg, #4c67a0, #9db1d9); }
  .telem-gauge-bg, .joint-bar-bg { background: #1c222e; }
  .switch input:checked + .switch-track { background: rgba(142,163,207,0.20); border-color: var(--accent); }
  .switch input:checked + .switch-track .switch-thumb { background: var(--accent); }
  .cmd-swatch { background: var(--muted-2) !important; }
  .cmd-card { border-color: var(--border) !important; }
  .cmd-card:hover { border-color: var(--accent) !important; background: rgba(142,163,207,0.07); transform: none; }
  .viewport-card, .telem-card, .joints-card { box-shadow: none; }
  .viewport-card { box-shadow: 0 10px 30px rgba(0,0,0,0.40); }
  .telem-card:hover { transform: none; }
  .stage-overlay { box-shadow: none; }
  .stage-overlay, .stage-overlay.idle { color: var(--text); }
  .stage-dot, .stage-overlay.idle .stage-dot { background: var(--accent); }
  .vp-badge { background: rgba(142,163,207,0.10); border-color: rgba(142,163,207,0.28); color: var(--accent); }
  .plan-step.RUNNING, .plan-step.REPLANNING { animation: none; box-shadow: 0 0 0 1px rgba(142,163,207,0.35); }
  .live-tag { display: inline-flex; align-items: center; gap: 6px; color: var(--green); }
  .live-dot, .feed-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--green); margin-right: 6px; vertical-align: 1px; }
  .live-tag .live-dot { margin-right: 0; }
  .feed-dot.off { background: var(--red); }
  .stack-chip { color: var(--muted); background: var(--card); border-color: var(--border-soft); font-weight: 500; }
  footer { background: var(--surface); border-top-color: var(--border-soft); }
  ::-webkit-scrollbar-thumb { background: #262e3d; }
  html { scrollbar-color: #262e3d transparent; }

  /* ── Header: brand, then the small pills right next to it (grouped, labelled, uniform) ── */
  header { min-height: 64px; flex-wrap: wrap; justify-content: flex-start; gap: 10px 40px; padding: 10px 26px; }
  .logo { gap: 18px; }
  .logo-ntt { height: 27px; width: auto; display: block; }
  .logo-sep { width: 1px; height: 32px; background: var(--border); display: block; }
  .logo-text p.tagline { font-size: 0.72rem; color: var(--muted); margin-top: 3px; letter-spacing: 0.15px; white-space: nowrap; }
  .header-right { display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 8px 16px; }
  .hgroup { display: flex; align-items: center; gap: 6px; }
  .hlabel { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; color: var(--muted-2); margin-right: 2px; }
  .hdivider { width: 1px; height: 22px; background: var(--border); }
  header .status-pill, header .svc-pill { height: 26px; padding: 0 11px; gap: 7px; font-size: 0.7rem; font-weight: 500; border-radius: 999px; white-space: nowrap; letter-spacing: 0.1px; }
  header .pill-icon { display: none; }
  header .svc-pill .svc-state { display: none; }
  header .svc-pill.svc-offline .svc-dot { background: #7a5f68; }
  header .hlink { color: var(--muted); text-decoration: none; cursor: pointer; transition: color 0.15s, border-color 0.15s; }
  header .hlink:hover { color: var(--text); border-color: #333d50; }
  header .dot { width: 7px; height: 7px; flex-shrink: 0; }

  /* ── Status colours: dull green = running / done, dull red = offline / failed, coral = hard reset ── */
  .svc-pill.svc-online { color: var(--ok); border-color: rgba(134,179,157,0.38); background: rgba(134,179,157,0.09); }
  .svc-pill.svc-online .svc-dot { background: var(--ok); }
  .svc-pill.svc-offline { color: var(--bad); border-color: rgba(196,128,127,0.38); background: rgba(196,128,127,0.09); }
  header .svc-pill.svc-offline .svc-dot, .svc-pill.svc-offline .svc-dot { background: var(--bad); }
  header .status-pill .dot { background: var(--ok); }
  header .status-pill .dot.busy { background: var(--amber); }
  header .status-pill .dot.off { background: var(--bad); animation: none; }
  .live-dot { background: var(--ok); } .live-tag { color: var(--ok); }
  /* smaller stage pill at the bottom of the viewport */
  .stage-overlay { bottom: 10px; padding: 3px 11px; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.3px; gap: 6px; background: rgba(9,11,15,0.70); }
  .stage-overlay .stage-dot { width: 5px; height: 5px; }
  .stage-overlay.idle .stage-dot { background: var(--ok); }
  .btn-danger { color: var(--coral); border: 1px solid rgba(224,112,95,0.55); background: rgba(224,112,95,0.06); }
  .btn-danger:hover:not(:disabled) { background: rgba(224,112,95,0.16); border-color: var(--coral); }
  .plan-step.DONE { border-color: rgba(134,179,157,0.45); }
  .badge-DONE, .plan-status-SUCCESS { background: rgba(134,179,157,0.16); color: var(--ok); }
  .plan-step.FAILED, .plan-step.REJECTED { border-color: rgba(196,128,127,0.5); }
  .badge-FAILED, .badge-REJECTED, .plan-status-FAILED, .plan-status-REJECTED, .plan-status-GOAL_NOT_HELD { background: rgba(196,128,127,0.16); color: var(--bad); }
  .plan-step.ABORTED { border-color: rgba(196,128,127,0.25); }

  /* ── About button (top right) and its dialog ── */
  header { position: relative; }
  .about-btn { position: absolute; top: 18px; right: 26px; height: 28px; padding: 0 14px; border-radius: 999px; background: transparent;
    border: 1px solid var(--border); color: var(--text); font: 500 0.72rem 'Inter', sans-serif; letter-spacing: 0.2px; cursor: pointer;
    transition: border-color 0.15s, background 0.15s; display: inline-flex; align-items: center; gap: 7px; }
  .about-btn:hover { border-color: var(--accent); background: rgba(142,163,207,0.08); }
  .about-btn svg { width: 13px; height: 13px; stroke: var(--accent); }
  .about-veil { position: fixed; inset: 0; z-index: 50; background: rgba(5,7,10,0.72); display: none; align-items: center; justify-content: center; padding: 24px; }
  .about-veil.open { display: flex; }
  .about-card { width: min(760px, 100%); max-height: 86vh; overflow-y: auto; background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 26px 30px 22px; box-shadow: 0 24px 70px rgba(0,0,0,0.6); }
  .about-head { display: flex; align-items: center; gap: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border-soft); margin-bottom: 6px; }
  .about-head img { height: 24px; }
  .about-head h2 { font-size: 1.05rem; font-weight: 600; letter-spacing: 0.2px; }
  .about-head p { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
  .about-x { margin-left: auto; background: none; border: 1px solid var(--border); color: var(--muted); width: 28px; height: 28px; border-radius: 8px; cursor: pointer; font-size: 1rem; line-height: 1; }
  .about-x:hover { color: var(--text); border-color: var(--accent); }
  .about-card h3 { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted-2); font-weight: 600; margin: 18px 0 8px; }
  .about-card p, .about-card li { font-size: 0.8rem; line-height: 1.55; color: #c9cfdb; }
  .about-card ul { list-style: none; display: grid; gap: 6px; }
  .about-card li { display: grid; grid-template-columns: 150px 1fr; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--border-soft); }
  .about-card li b { color: var(--text); font-weight: 600; }
  .about-card ol { padding-left: 18px; display: grid; gap: 5px; }
  .about-credit { margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--border-soft); text-align: center; font-size: 0.72rem; color: var(--muted); letter-spacing: 0.3px; }
  .about-credit b { color: var(--text); font-weight: 600; }
</style>
</head>
<body>

<header>
  <button class="about-btn" onclick="openAbout()" title="About Physyk AI"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>About</button>
  <div class="logo">
    <img class="logo-ntt" src="ntt_data_logo.png" alt="NTT DATA">
    <span class="logo-sep"></span>
    <div class="logo-text">
      <h1>Physyk AI</h1>
      <p class="tagline">__STACK_LINE__</p>
    </div>
  </div>
  <div class="header-right">
    <div class="hgroup">
      <span class="hlabel">Mode</span>
      <div class="status-pill" id="rmpPill" title="Arm motion: NVIDIA RMPFlow with the PickPlaceController"><span>RMPFlow</span></div>
      <div class="status-pill" id="modePill" title="Whether instructions go through the Nemotron agentic orchestrator (decompose / guardrail / verify / retry) or straight to Isaac Sim as typed."><span class="pill-icon" id="modePillIcon"></span><span id="modePillText">Agentic</span></div>
      <div class="status-pill" id="perceptionPill" title="Where object positions come from: the overview camera only (colour segmentation + depth), or the simulator's own poses."><span class="pill-icon"></span><span id="perceptionPillText">Perception</span></div>
      <div class="status-pill"><div class="dot" id="statusDot"></div><span id="statusText">Sim idle</span></div>
    </div>
    <span class="hdivider"></span>
    <div class="hgroup svc-strip" title="Real process health of the three backing services. Offline means the service is not running at all, which is separate from the 'Send prompt to Agent' toggle.">
      <span class="hlabel">Services</span>
      <div class="svc-pill svc-offline" id="svc-isaac_sim"><span class="svc-dot"></span><span class="svc-label">Isaac Sim</span><span class="svc-state">—</span></div>
      <div class="svc-pill svc-offline" id="svc-nemotron"><span class="svc-dot"></span><span class="svc-label">Nemotron</span><span class="svc-state">—</span></div>
      <div class="svc-pill svc-offline" id="svc-cosmos"><span class="svc-dot"></span><span class="svc-label">Cosmos</span><span class="svc-state">—</span></div>
    </div>
    <span class="hdivider"></span>
    <div class="hgroup">
      <span class="hlabel">Logs</span>
      <a href="/reasoning/nemotron" target="_blank" class="status-pill hlink" title="Full Nemotron reasoning log">Nemotron</a>
      <a href="/reasoning/cosmos" target="_blank" class="status-pill hlink" title="Full Cosmos log, with the camera frames it was shown">Cosmos</a>
      <a href="/reasoning/perception" target="_blank" class="status-pill hlink" title="Camera detections: annotated frames, bounding boxes and estimated XYZ for every pre-pick detection and post-place check">Perception</a>
    </div>
  </div>
</header>

<div class="about-veil" id="aboutVeil" onclick="if(event.target===this)closeAbout()">
  <div class="about-card" role="dialog" aria-label="About Physyk AI">
    <div class="about-head">
      <img src="ntt_data_logo.png" alt="NTT DATA">
      <div><h2>Physyk AI</h2><p>Plain-English pick-and-place for a simulated Franka Panda arm</p></div>
      <button class="about-x" onclick="closeAbout()" aria-label="Close">&times;</button>
    </div>

    <h3>What it does</h3>
    <p>You describe a task in plain language. Physyk plans the moves, finds the cubes by looking at camera images, runs the pick-and-place in NVIDIA Isaac Sim, and checks the real outcome before it reports success. A second AI model gives an independent visual opinion on every step.</p>

    <h3>How a request flows</h3>
    <ol>
      <li><b>Plan.</b> "Stack ..." is planned by a local program from the cube positions. Anything else goes to Nemotron, which splits it into goals once and then proposes one move at a time against the live scene.</li>
      <li><b>Guard.</b> Plain rules refuse any target outside the arm's safe reach.</li>
      <li><b>Move.</b> NVIDIA's pick-and-place controller drives the arm through RMPFlow motion planning.</li>
      <li><b>Verify.</b> The cube's real position is checked (simulator pose, or a fresh camera look in camera-only mode); Cosmos checks the picture in the background.</li>
      <li><b>Adapt.</b> After every step the plan is re-made from the live scene, so a dropped or moved cube is handled.</li>
    </ol>

    <h3>Components</h3>
    <ul>
      <li><b>Isaac Sim 6.0.1</b><span>Physics at 60 Hz, overview and wrist cameras, runs headless in Docker.</span></li>
      <li><b>Nemotron 3.5 Lightning 30B</b><span>Plans non-stacking requests (NVIDIA hosted API, thinking off, streamed).</span></li>
      <li><b>Stack planner</b><span>Local and deterministic: decides the base cube, the order, and where blocking cubes go.</span></li>
      <li><b>Cosmos 3 Nano Reasoner</b><span>Local NIM that confirms each step visually; evidence only, it never steers the arm.</span></li>
      <li><b>Camera perception</b><span>Colour masks and depth locate the cubes in about 20 ms (the "Pure perception" switch).</span></li>
      <li><b>Backend and GUI</b><span>FastAPI on port 7860, talking to the simulator on port 8100.</span></li>
    </ul>

    <h3>Using it</h3>
    <ul>
      <li><b>Instruction box</b><span>Type a task, for example "put the red cube in the tray" or "stack the cubes". Several steps can be chained with "then".</span></li>
      <li><b>Send prompt to Agent</b><span>On: planned, guarded and verified. Off: the text goes straight to the simulator.</span></li>
      <li><b>Pure perception</b><span>On: the arm works from the camera alone and boxes are drawn on the live view. Off: simulator positions.</span></li>
      <li><b>Quick actions</b><span>Randomize the cubes, return the arm home, or reset the scene. Hard Reset stops everything.</span></li>
      <li><b>Logs</b><span>The Nemotron, Cosmos and Perception pages show the full reasoning and the exact frames the models saw.</span></li>
    </ul>

    <h3>Good to know</h3>
    <p>This is a simulation with three coloured cubes and a tray on a known table. Cubes are reachable between about 0.38 m and 0.66 m from the arm's base. The hand always points straight down and the grasp is timed rather than closed-loop, so very long sideways carries may overshoot.</p>

    <div class="about-credit">Built by the <b>India Innovation Centre, NTT DATA</b></div>
  </div>
</div>
<script>
  function openAbout() { document.getElementById('aboutVeil').classList.add('open'); }
  function closeAbout() { document.getElementById('aboutVeil').classList.remove('open'); }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeAbout(); });
  if (location.hash === '#about') openAbout();
</script>

<div class="layout">
  <!-- VIEWPORT & TELEMETRY -->
  <div class="viewport-panel">
    <!-- Camera Viewport Card -->
    <div class="viewport-card">
      <div class="viewport-header">
        <div class="viewport-title">
          <span><svg class="icon" viewBox="0 0 24 24"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/></svg> Live Viewport</span>
          <select id="cameraSelector" class="vp-select">
            <option value="scene">3rd Person (Isometric Overview)</option>
            <option value="wrist">Wrist Camera (Gripper Close-up)</option>
          </select>
          <span class="vp-badge" id="feedStatus">HTTP Stream</span>
        </div>
        <span class="fps-readout"><span class="live-tag"><i class="live-dot"></i>LIVE</span><span id="fpsBadge">— FPS</span></span>
      </div>
      <div class="stream-container">
        <img class="stream-video" id="liveStreamImg" src="" alt="Isaac Sim Live Viewport" />
        <div class="stage-overlay idle" id="stageOverlay"><span class="stage-dot"></span>STAGE: READY</div>
        <div id="streamLoadingOverlay" style="
            position:absolute; inset:0; display:flex; flex-direction:column;
            align-items:center; justify-content:center; background:rgba(3,5,7,0.94);
            color:var(--accent); font-size:0.9rem; font-weight:700; gap:12px; z-index:5;">
          <div><svg viewBox="0 0 24 24" style="width:32px;height:32px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;"><circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M3 12h3M18 12h3M4.9 19.1L7 17M17 7l2.1-2.1"/></svg></div>
          <div>Isaac Sim Initializing...</div>
          <div style="font-size:0.75rem;color:var(--muted);">Camera feed will appear when ready</div>
        </div>
      </div>
    </div>

    <!-- Telemetry Cards -->
    <div class="telemetry-row">
      <div class="telem-card">
        <div class="telem-label"><span>Render Loop</span><span class="telem-tag live">LIVE</span></div>
        <div class="telem-val" id="fpsVal" style="color:var(--green)">— <span class="telem-unit">FPS</span></div>
      </div>
      <div class="telem-card">
        <div class="telem-label"><span>Physics Timestep</span><span class="telem-tag cfg">CONFIG</span></div>
        <div class="telem-val" style="color:var(--text)">16.7 <span class="telem-unit">ms · 60 Hz</span></div>
      </div>
      <div class="telem-card">
        <div class="telem-label"><span>Gripper Width</span><span class="telem-tag live">LIVE</span></div>
        <div class="telem-val" id="gripperVal">0.040 <span class="telem-unit">m</span></div>
        <div class="telem-gauge-bg"><div class="telem-gauge-fill" id="gripperGauge" style="width:100%"></div></div>
      </div>
      <div class="telem-card">
        <div class="telem-label"><span>End-Effector (m)</span><span class="telem-tag live">LIVE</span></div>
        <div class="telem-val telem-xyz mono" id="eePosVal">
          <div><span>X</span>0.450</div><div><span>Y</span>0.000</div><div><span>Z</span>0.400</div>
        </div>
      </div>
      <div class="telem-card">
        <div class="telem-label"><span>Current Stage</span><span class="telem-tag live">LIVE</span></div>
        <div class="telem-val" id="stateLabel" style="font-size:0.88rem; color:var(--accent)">READY</div>
      </div>
    </div>

    <!-- Joints Monitoring -->
    <div class="joints-card">
      <div class="joints-title">7-DOF Franka Panda Joint Angles (Radians) — Live</div>
      <div class="joint-grid" id="jointBars">
        <!-- Rendered via JS -->
      </div>
    </div>
  </div>

  <!-- SIDEBAR CONTROLS -->
  <div class="sidebar">
    <!-- Prompt Panel -->
    <div class="prompt-panel">
      <div class="section-title">Physical AI Instruction</div>
      <textarea id="promptInput" class="prompt-box" placeholder="e.g., Pick up the red cube and place it into the tray..."></textarea>
      <div class="btn-row">
        <button class="btn btn-primary" id="btnExecute" onclick="sendInstruction()"><svg class="icon" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg> Execute in Sim</button>
        <button class="btn btn-danger" id="btnHardReset" onclick="sendHardReset()" title="Immediately stop any running prompt or agentic plan, clear it, send the arm home, and return every cube to its default staging spot."><svg class="icon" viewBox="0 0 24 24"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg> Hard Reset</button>
      </div>
      <div class="agent-switch-row" title="When on, prompts are decomposed/guardrailed/verified/retried by the Nemotron agentic orchestrator. When off, prompts go straight to Isaac Sim exactly as typed — no agent involved.">
        <label class="switch">
          <input type="checkbox" id="agentSwitch" onchange="toggleAgent()">
          <span class="switch-track"><span class="switch-thumb"></span></span>
        </label>
        <span class="agent-switch-label" id="agentSwitchLabel">Send prompt to Agent</span>
      </div>
      <div class="agent-switch-row" id="perceptionSwitchRow" title="Off: the arm goes to the simulator's exact object positions (deterministic). On: PURE PERCEPTION — object positions come only from the overview camera (color segmentation + depth); the arm never sees simulator coordinates, and nothing falls back to them. If the camera cannot see an object the task is rejected. Takes effect on the next pick-and-place.">
        <label class="switch">
          <input type="checkbox" id="perceptionSwitch" onchange="togglePerception()">
          <span class="switch-track"><span class="switch-thumb"></span></span>
        </label>
        <span class="agent-switch-label" id="perceptionSwitchLabel">Deterministic — sim positions</span>
      </div>
    </div>

    <!-- Quick Tasks -->
    <div class="quick-cmds">
      <div class="section-title">Quick Action Tasks</div>
      <div class="cmd-grid">
        <div class="cmd-card" onclick="sendRandomize()" style="border-color: rgba(179,169,143,0.35);" title="Places all 3 cubes at genuinely random positions (np.random, not a fixed list) anywhere within the arm's reach, clear of the tray.">
          <span class="cmd-swatch" style="background:#b3a98f;"></span>
          <span><svg class="icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/></svg> Randomize Cube Positions</span>
        </div>
        <div class="cmd-card" onclick="runQuick('Return robot arm to home configuration')" title="Moves only the robot arm — cube positions are left exactly where they are.">
          <span class="cmd-swatch" style="background:#7e879a;"></span>
          <span>Return to Home Pose (arm only)</span>
        </div>
        <div class="cmd-card" onclick="sendReset()" style="border-color: rgba(134,165,200,0.3);">
          <span class="cmd-swatch" style="background:#86a5c8;"></span>
          <span>Full Reset — Cubes to Fixed Spots + Home</span>
        </div>
      </div>
      <p style="font-size:0.72rem; color:var(--muted); margin-top:10px; line-height:1.5;">Describe the task in plain language, for example: pick up the green cube and place it in the tray.</p>
    </div>

    <!-- Real Agentic Plan (Nemotron decompose -> guardrail -> dispatch -> verify -> replan) -->
    <div class="plan-panel">
      <div class="section-title"><svg class="icon" viewBox="0 0 24 24"><circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="13" r="2.4"/><circle cx="6" cy="19" r="2.4"/><circle cx="18" cy="19" r="2.4"/><path d="M7.7 7.5L10.5 11.5M16.3 7.5L13.5 11.5M10.7 14.8L7.3 17.7M13.3 14.8L16.7 17.7"/></svg> Agent Plan</div>
      <div id="planBox">
        <div class="plan-empty">No plan running — type an instruction above.</div>
      </div>
    </div>

    <!-- Cosmos-Reason2 shadow-mode perception (cosmos_integration.md Step 6) -->
    <div class="plan-panel">
      <div class="section-title"><svg class="icon" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="2.6"/></svg> Vision Perception</div>
      <div id="perceptionBox">
        <div class="plan-empty">No perception snapshot yet — run an instruction.</div>
      </div>
    </div>

    <!-- Direct API & Camera Streams Directory -->
    <div class="quick-cmds">
      <div class="section-title"><svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 010 18M12 3a14 14 0 000 18"/></svg> Live API &amp; Stream Directory</div>
      <div style="display:flex; flex-direction:column; gap:6px;">
        <div class="api-row"><span>3rd Person Isometric Stream</span><a data-path="/camera/stream" target="_blank">Open <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>In-Hand Wrist Camera Stream</span><a data-path="/camera/wrist.stream" target="_blank">Open <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>Live Joint State JSON</span><a data-path="/state" target="_blank">View /state <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>System Health JSON</span><a data-path="/health" target="_blank">View /health <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>Vision Perception JSON</span><a data-path="/perception" target="_blank">View /perception <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
      </div>
    </div>

    <!-- Live Execution Logs -->
    <div class="logs-panel">
      <div class="section-title">Live Activity Log</div>
      <div class="logs-box" id="logBox">
        <div class="log-entry"><span class="log-time">[Init]</span> <span class="log-msg SUCCESS">Connecting to Physyk backend...</span></div>
      </div>
    </div>
  </div>
</div>

<footer>
  <span class="stack-chip">Isaac Sim 6.0.1 (Docker)</span>
  <span class="stack-chip">PhysX</span>
  <span class="stack-chip">RMPFlow + PickPlaceController</span>
  <span class="stack-chip">Franka Panda 7-DOF</span>
  <span class="stack-chip">__NEMOTRON_CHIP__</span>
  <span class="stack-chip">Cosmos 3 Nano Reasoner (NIM)</span>
  <span class="stack-chip">Colour + depth perception</span>
</footer>

<script>
  const bp = window.location.pathname.replace(/\\/$/, '');
  // The API/stream directory links are rendered with a relative data-path — resolved here
  // against the real base path (accounts for being served behind a reverse proxy, same as
  // every fetch() below already does via `bp`). Setting bare "/state"-style hrefs directly
  // 404'd through the proxy, since only this base path is proxy-aware.
  document.querySelectorAll('.api-row a[data-path]').forEach(a => { a.href = bp + a.dataset.path; });
  const liveFeed = document.getElementById('liveStreamImg');

  const jointGrid = document.getElementById('jointBars');
  for (let i = 1; i <= 7; i++) {
    jointGrid.innerHTML += `
      <div class="joint-row">
        <div class="joint-name">J${i}</div>
        <div class="joint-bar-bg">
          <div class="joint-bar-fill" id="jbar_${i}" style="width: 50%"></div>
        </div>
        <div class="joint-val mono" id="jval_${i}">0.000</div>
      </div>
    `;
  }

  // Log rendering is driven by the backend's authoritative log (sim_state["log"], polled via
  // /state) rather than optimistic client-side guesses, so what's shown always reflects what
  // the server actually did (or failed to do) — not what the button click assumed would happen.
  let shownLogCount = 0;
  function renderLog(entries) {
    if (!entries || entries.length <= shownLogCount) return;
    const box = document.getElementById('logBox');
    const fresh = entries.slice(shownLogCount);
    for (const e of fresh) {
      box.innerHTML += `<div class="log-entry"><span class="log-time">[${e.t}]</span> <span class="log-msg ${e.level}">${e.msg}</span></div>`;
    }
    shownLogCount = entries.length;
    box.scrollTop = box.scrollHeight;
  }

  // Renders the real agent plan (nemo_agent.run()'s on_plan_update snapshots via /state.plan)
  // — every field here reflects an actual decompose/guardrail/dispatch/verify/replan
  // transition the orchestrator went through, not a client-side guess at what's happening.
  const STATUS_ICON = {
    PENDING: '<svg class="icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3" fill="currentColor" stroke="none"/></svg>',
    RUNNING: '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M7 4l13 8-13 8z"/></svg>',
    RETRYING: '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M4 12a8 8 0 0113.6-5.7M20 12a8 8 0 01-13.6 5.7"/><path d="M17 3v4h-4M7 21v-4h4"/></svg>',
    REPLANNING: '<svg class="icon-sm" viewBox="0 0 24 24"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="13" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M7.7 7.3L10.5 11.3M16.3 7.3L13.5 11.3M10.7 14.6L7.3 17.5M13.3 14.6L16.7 17.5"/></svg>',
    DONE: '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M4 12l5 6L20 5"/></svg>',
    FAILED: '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M5 5l14 14M19 5L5 19"/></svg>',
    REJECTED: '<svg class="icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M6 6l12 12"/></svg>',
    ABORTED: '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M6 12h12"/></svg>',
  };
  // Cosmos-Reason2 visual step-verification evidence (cosmos_integration.md Step 7) — a
  // step goes "green" because the model looked at the result, not just because the
  // planner returned. visual_enforced=false means the check couldn't meaningfully run
  // (Cosmos down/timeout, or a skipped/no-motion step) — shown as NOT CHECKED, not hidden,
  // so the UI never implies a check happened when it didn't.
  function renderVisualEvidence(step) {
    if (step.visual_observed == null) return '';
    const enforced = !!step.visual_enforced;
    const verified = !!step.visual_verified;
    let color, label;
    if (!enforced) { color = 'var(--muted)'; label = 'NOT CHECKED'; }
    else if (verified) { color = '#86b39d'; label = 'VERIFIED'; }
    else { color = '#b3a98f'; label = 'VISUAL CHECK DISAGREES'; }
    const conf = (enforced && step.visual_confidence != null) ? ` · conf ${step.visual_confidence}` : '';
    return `<div class="plan-step-detail" style="color:${color}; margin-top:4px;">
      <span style="font-weight:700; letter-spacing:0.4px;">${label}${conf}</span> — "${step.visual_observed}"
    </div>`;
  }

  // Cosmos-Reason2 PRE-dispatch reasoning (target visible / obstacle / feasibility /
  // approach) — the "is this action feasible, what path looks right" judgment made BEFORE
  // the arm moves, distinct from renderVisualEvidence's AFTER-the-fact check. Purely
  // informational (never gates dispatch) — see _attach_cosmos_assessment in the orchestrator.
  // Full paragraphs now live on each agent's own reasoning page (see reasoning_dashboard.py)
  // — this panel shows the short, scannable facts only, with a link out to the full text
  // instead of inlining it. Relative paths on this same origin — no separate port to expose.
  const COSMOS_LOG_URL = '/reasoning/cosmos';
  const NEMOTRON_LOG_URL = '/reasoning/nemotron';

  function renderCosmosAssessment(step) {
    const a = step.cosmos_assessment;
    if (!a) return '';
    if (a.error) {
      return `<div class="plan-step-detail" style="color:var(--muted); margin-top:4px;">COSMOS: ${a.error}</div>`;
    }
    const feasColor = a.feasibility === 'high' ? '#7fb4c9' : (a.feasibility === 'medium' ? '#b3a98f' : (a.feasibility === 'low' ? '#b88a93' : 'var(--muted)'));
    const tv = a.target_visible === true ? 'yes' : (a.target_visible === false ? 'no' : 'n/a');
    const dv = a.destination_visible === true ? 'yes' : (a.destination_visible === false ? 'no' : (a.destination_visible === null ? 'n/a' : 'n/a'));
    let html = `<div class="plan-step-detail" style="margin-top:4px;">
      <span style="font-weight:700; letter-spacing:0.4px; color:#8ea3cf;">COSMOS 3 NANO REASONER</span>
      &nbsp;target ${tv} &nbsp;dest ${dv}
      &nbsp;feasibility <span style="color:${feasColor}; font-weight:700;">${(a.feasibility || '?').toUpperCase()}</span>`;
    if (a.approach) html += `&nbsp;· approach: ${a.approach}`;
    if (a.obstacle && a.obstacle.toLowerCase() !== 'none') html += `&nbsp;· obstacle: ${a.obstacle}`;
    if (a.reasoning) html += `&nbsp;· <a href="${COSMOS_LOG_URL}" target="_blank" style="color:#7fb4c9;">View full reasoning</a>`;
    html += '</div>';
    return html;
  }

  // Nemotron's own diagnosis text when it decided retry/replan on a real failure — one-line
  // summary here (the decision it made), full chain-of-thought on its own dashboard.
  function renderPlannerReasoning(step) {
    if (!step.planner_reasoning) return '';
    const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    // a move proposed step by step carries the model's one-sentence reason; a failure decision carries its reasoning text
    const text = step.raw_answer ? esc(step.planner_reasoning) : 'Nemotron diagnosed this failure and chose how to proceed.';
    return `<div class="plan-step-detail" style="color:var(--muted); margin-top:4px;">
      ${step.raw_answer ? '<span style="color:#8ea3cf; font-weight:600;">NEMOTRON</span> ' : ''}${text}
      <a href="${NEMOTRON_LOG_URL}" target="_blank" style="color:#8ea3cf;">View full reasoning</a>
    </div>`;
  }

  let lastPlanJson = null;
  function renderPlan(plan, agentEnabled) {
    const box = document.getElementById('planBox');
    const planJson = JSON.stringify(plan) + '|' + agentEnabled;
    if (planJson === lastPlanJson) return;
    lastPlanJson = planJson;

    // Agent Plan panel goes on hold while the "Send prompt to Agent" switch is off — this
    // deliberately overrides any lingering plan from before the switch was flipped, rather
    // than leaving a stale plan visible while prompts are actually bypassing the agent.
    if (!agentEnabled) {
      box.innerHTML = '<div class="plan-empty">Agent is off — prompts go straight to Isaac Sim. Plan on hold.</div>';
      return;
    }

    if (!plan) {
      box.innerHTML = '<div class="plan-empty">No plan running — type an instruction above.</div>';
      return;
    }
    const overall = plan.overall || 'PLANNING';
    const statusClass = 'plan-status-' + overall + (plan.active ? ' active' : '');
    let html = `<div class="plan-instruction"><span class="plan-status ${statusClass}">${overall}</span>"${plan.instruction}"</div>`;
    // Nemotron's own chain-of-thought for *why* it split the instruction into these steps —
    // shown once, above the step list, so "the planner is thinking" is visible up front.
    if (plan.planning_reasoning) {
      const stepCount = (plan.steps || []).length;
      const stepwise = Array.isArray(plan.milestones);
      html += `<div class="plan-step-detail" style="color:var(--muted); margin:6px 0 10px; padding:8px; border-left:2px solid #8ea3cf; background:rgba(142,163,207,0.06);">
        <span style="font-weight:700; color:#8ea3cf;">NEMOTRON PLANNING</span> ${stepwise
          ? `split the request into ${plan.milestones.length} goal${plan.milestones.length === 1 ? '' : 's'}; moves are planned one at a time (${stepCount} so far).`
          : `decomposed the request into ${stepCount} step${stepCount === 1 ? '' : 's'}.`}
        <a href="${NEMOTRON_LOG_URL}" target="_blank" style="color:#8ea3cf;">View full reasoning</a>
      </div>`;
    }
    // Live token stream of the model's answer while it is being planned (a short JSON line, streamed as it is generated).
    if (plan.planner_stream && plan.active) {
      html += `<div class="plan-step-detail" style="color:var(--muted); margin:0 0 10px; padding:6px 8px; font-family:monospace; font-size:11px; border-left:2px solid #8ea3cf; opacity:0.85;">${plan.planner_stream.replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]))}</div>`;
    }
    const escHtml = (t) => t.replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    if (plan.answer && !plan.active) {
      const bad = ['REJECTED', 'FAILED', 'PARTIAL'].includes(overall);
      html += `<div class="plan-step-detail" style="margin:8px 0 10px; padding:9px 11px; font-weight:600; line-height:1.45; color:${bad ? '#e3bcbb' : '#bfdccb'}; border-left:3px solid ${bad ? '#c4807f' : '#86b39d'}; background:${bad ? 'rgba(196,128,127,0.14)' : 'rgba(134,179,157,0.11)'};">${escHtml(String(plan.answer))}</div>`;
    }
    html += '<div class="plan-steps">';
    for (const step of (plan.steps || [])) {
      const status = step.status || 'PENDING';
      html += `<div class="plan-step ${status}">
        <div class="plan-step-idx">${step.index}</div>
        <div class="plan-step-body">
          <div class="plan-step-text">${step.text || ''}</div>
          ${step.refused
            ? `<div class="plan-step-detail" style="color:#e3bcbb; margin-top:6px; padding:8px 10px; border-left:3px solid #c4807f; background:rgba(196,128,127,0.14); font-weight:600; line-height:1.45;">Failed: ${escHtml(String(step.detail || '').replace(/^refused by the robot: /, ''))}</div>`
            : (step.detail ? `<div class="plan-step-detail">${step.detail}</div>` : '')}
          ${renderPlannerReasoning(step)}
          ${renderCosmosAssessment(step)}
          ${renderVisualEvidence(step)}
        </div>
        <div class="plan-step-badge badge-${status}">${STATUS_ICON[status] || ''} ${status}</div>
      </div>`;
    }
    html += '</div>';
    box.innerHTML = html;
  }

  // Renders the real Cosmos-Reason2 shadow-mode snapshot (/state.perception, mirrors Isaac
  // Sim's own /perception) — stage pose vs Cosmos's deprojected estimate per object, one
  // grounding call per dispatched pick/place instruction. "used" is always "stage" unless
  // PERCEPTION_MODE=vision is set server-side AND that object's estimate passed the sanity
  // gate; this panel just displays whatever the backend actually did, never assumes.
  const PERC_LABELS = { red_cube: 'Red Cube', blue_cube: 'Blue Cube', green_cube: 'Green Cube', target_tray: 'Tray' };
  let lastPerceptionJson = null;
  function renderPerception(perc) {
    const box = document.getElementById('perceptionBox');
    const pill = document.getElementById('perceptionPill');
    const pillText = document.getElementById('perceptionPillText');
    const json = JSON.stringify(perc);
    if (json === lastPerceptionJson) return;
    lastPerceptionJson = json;

    const mode = perc ? perc.mode : null;
    if (mode && !perceptionSwitchUserTouched) setPerceptionSwitchUI(mode === 'pure');
    if (mode) {
      pill.className = 'status-pill mode-' + mode;
      pillText.textContent = ({pure: 'Camera only', vision: 'Camera + sim', shadow: 'Sim positions', stage: 'Sim positions'})[mode] || mode;
    } else {
      pill.className = 'status-pill';
      pillText.textContent = 'Perception';
    }

    if (!perc || !perc.objects || Object.keys(perc.objects).length === 0) {
      box.innerHTML = `<div class="plan-empty">No perception snapshot yet — run an instruction.${perc && perc.error ? ' (' + perc.error + ')' : ''}</div>`;
      return;
    }

    let html = '<div class="plan-steps">';
    for (const [key, obj] of Object.entries(perc.objects)) {
      const label = PERC_LABELS[key] || key;
      const delta = obj.delta_mm;
      let deltaHtml, deltaColor;
      if (delta == null) {
        deltaHtml = 'no estimate'; deltaColor = 'var(--muted)';
      } else {
        deltaColor = delta < 20 ? '#7fb4c9' : (delta < 80 ? '#b3a98f' : '#b88a93');
        deltaHtml = delta.toFixed(1) + ' mm delta';
      }
      html += `<div class="plan-step DONE">
        <div class="plan-step-body">
          <div class="plan-step-text">${label} <span style="color:var(--muted); font-weight:400;">— driving pose: ${obj.used}</span></div>
          <div class="plan-step-detail" style="color:${deltaColor};">${deltaHtml}</div>
        </div>
      </div>`;
    }
    html += '</div>';
    if (perc.latency_ms != null) {
      const det = perc.detector === 'color' ? 'Color-segmentation' : 'Cosmos grounding';
      html += `<div class="plan-step-detail" style="margin-top:8px;">${det} latency: ${perc.latency_ms.toFixed(0)} ms</div>`;
    }
    box.innerHTML = html;
  }

  async function pollState() {
    try {
      const res = await fetch(bp + '/state');
      if (res.ok) {
        const data = await res.json();

        const modePill = document.getElementById('modePill');
        const modeIcon = document.getElementById('modePillIcon');
        const modeText = document.getElementById('modePillText');
        if (data.agentic_mode) {
          modePill.className = 'status-pill mode-agentic';
          modeIcon.innerHTML = '<svg class="icon-sm" viewBox="0 0 24 24"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="13" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M7.7 7.3L10.5 11.3M16.3 7.3L13.5 11.3M10.7 14.6L7.3 17.5M13.3 14.6L16.7 17.5"/></svg>';
          modeText.textContent = 'Agentic';
        } else {
          modePill.className = 'status-pill mode-direct';
          modeIcon.innerHTML = '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg>';
          modeText.textContent = 'Direct';
        }

        // 4-service status panel — reflects real process health (poll_service_status in
        // Python), independent of the agent toggle: a service can be RUNNING while the
        // toggle is OFF (it's just not being dispatched to), and OFFLINE services show
        // clearly rather than the panel silently omitting them.
        if (data.service_status) {
          const svcLabels = {isaac_sim: 'Isaac Sim', nemotron: 'Nemotron', cosmos: 'Cosmos'};
          for (const [key, label] of Object.entries(svcLabels)) {
            const el = document.getElementById('svc-' + key);
            if (!el) continue;
            const online = !!data.service_status[key];
            el.className = 'svc-pill ' + (online ? 'svc-online' : 'svc-offline');
            el.querySelector('.svc-label').textContent = label;
            el.querySelector('.svc-state').textContent = online ? 'ONLINE' : 'OFFLINE';
          }
        }

        const stage = (data.stage || 'READY').toUpperCase();
        const stageEl = document.getElementById('stageOverlay');
        stageEl.innerHTML = '<span class="stage-dot"></span>STAGE: ' + stage;
        stageEl.className = 'stage-overlay' + (data.busy ? '' : ' idle');
        document.getElementById('stateLabel').textContent = data.stage || 'READY';

        const fps = (data.fps || 0).toFixed(1);
        document.getElementById('fpsBadge').textContent = fps + ' FPS';
        document.getElementById('fpsVal').innerHTML = fps + ' <span class="telem-unit">FPS</span>';

        const gripper = (data.gripper != null ? data.gripper : 0.04);
        document.getElementById('gripperVal').innerHTML = gripper.toFixed(3) + ' <span class="telem-unit">m</span>';
        const gripperPct = Math.max(0, Math.min(100, (gripper / 0.04) * 100));
        document.getElementById('gripperGauge').style.width = gripperPct + '%';

        if (data.ee_pos && data.ee_pos.length >= 3) {
          const [x, y, z] = data.ee_pos;
          document.getElementById('eePosVal').innerHTML =
            `<div><span>X</span>${x.toFixed(3)}</div><div><span>Y</span>${y.toFixed(3)}</div><div><span>Z</span>${z.toFixed(3)}</div>`;
        }

        const dot = document.getElementById('statusDot');
        const btn = document.getElementById('btnExecute');
        const simOffline = data.service_status && data.service_status.isaac_sim === false;
        if (simOffline) {
            dot.className = 'dot off';
            document.getElementById('statusText').textContent = 'Sim offline';
            btn.disabled = true;
        } else if (data.busy) {
            dot.className = 'dot busy';
            document.getElementById('statusText').textContent = 'Sim busy';
            btn.disabled = true;
        } else {
            dot.className = 'dot';
            document.getElementById('statusText').textContent = 'Sim idle';
            btn.disabled = false;
        }

        if (data.joints && data.joints.length >= 7) {
          for (let i = 0; i < 7; i++) {
            const rad = data.joints[i];
            const pct = Math.max(0, Math.min(100, ((rad + 3.14) / 6.28) * 100));
            const fill = document.getElementById('jbar_' + (i + 1));
            const val = document.getElementById('jval_' + (i + 1));
            if (fill) fill.style.width = pct + '%';
            if (val) val.textContent = rad.toFixed(3);
          }
        }

        renderLog(data.log);
        renderPlan(data.plan, data.agentic_mode);
        renderPerception(data.perception);

        // Keep the switch in sync with the server's live state — covers another tab/user
        // flipping it, and the initial page load reflecting whatever it actually started as.
        if (!agentSwitchUserTouched) {
          const sw = document.getElementById('agentSwitch');
          sw.checked = !!data.agentic_mode;
          document.getElementById('agentSwitchLabel').textContent =
            data.agentic_mode ? 'Send prompt to Agent' : 'Agent off — direct to Isaac Sim';
          document.querySelector('.agent-switch-row').classList.toggle('on', !!data.agentic_mode);
        }
      }
    } catch(e) {}
  }

  // Camera feed switcher logic
  const feedStatus = document.getElementById('feedStatus');
  let currentFeed = "scene";

  // MJPEG stream — single persistent connection, ~30 FPS, zero polling overhead
  function connectMJPEGStream() {
      let endpoint = '/camera/stream';
      if (currentFeed === 'wrist') endpoint = '/camera/wrist.stream';

      liveFeed.src = bp + endpoint;
      liveFeed.onload = () => {
          const names = {
            scene: 'Overview', wrist: 'Wrist'
          };
          feedStatus.innerHTML = '<span class="feed-dot on"></span>Live (' + (names[currentFeed] || currentFeed) + ')';
          const overlay = document.getElementById('streamLoadingOverlay');
          if (overlay) overlay.style.display = 'none';
      };
      liveFeed.onerror = () => {
          feedStatus.innerHTML = '<span class="feed-dot off"></span>Connecting...';
          const overlay = document.getElementById('streamLoadingOverlay');
          if (overlay) overlay.style.display = 'flex';
          setTimeout(connectMJPEGStream, 2500); // Retry every 2.5s
      };
  }
  connectMJPEGStream();

  // Re-connect when camera selector changes
  document.getElementById('cameraSelector').addEventListener('change', (e) => {
      currentFeed = e.target.value;
      connectMJPEGStream();
  });

  setInterval(pollState, 150);
  pollState();

  async function sendInstruction() {
    const input = document.getElementById('promptInput');
    const instruction = input.value.trim();
    if (!instruction) return;
    const btn = document.getElementById('btnExecute');
    btn.disabled = true;
    try {
      const res = await fetch(bp + '/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction })
      });
      if (!res.ok) btn.disabled = false;
      // Confirmation/failure appears in the log via the next /state poll — no optimistic
      // client-side guess here, since the backend is the only one that knows what happened.
    } catch(e) {
      btn.disabled = false;
    }
  }

  function runQuick(text) {
    document.getElementById('promptInput').value = text;
    sendInstruction();
  }

  async function sendReset() {
    try {
      await fetch(bp + '/reset', { method: 'POST' });
    } catch(e) {}
  }

  // True for a brief window right after the user clicks the switch, so the next couple of
  // /state polls (which may still report the pre-toggle value while the request is in
  // flight) don't visually snap it back before the server confirms the real new state.
  let agentSwitchUserTouched = false;
  let perceptionSwitchUserTouched = false;
  function setPerceptionSwitchUI(vision) {
    document.getElementById('perceptionSwitch').checked = vision;
    document.getElementById('perceptionSwitchLabel').textContent =
      vision ? 'Pure perception — camera only' : 'Deterministic — sim positions';
    document.getElementById('perceptionSwitchRow').classList.toggle('on', vision);
  }
  async function togglePerception() {
    const vision = document.getElementById('perceptionSwitch').checked;
    perceptionSwitchUserTouched = true;
    setPerceptionSwitchUI(vision);
    try {
      const res = await fetch(bp + '/perception_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vision })
      });
      if (!res.ok) setPerceptionSwitchUI(!vision);   // switch failed — show the real state
    } catch(e) {
      setPerceptionSwitchUI(!vision);
    } finally {
      perceptionSwitchUserTouched = false;
    }
  }

  async function toggleAgent() {
    const sw = document.getElementById('agentSwitch');
    const enabled = sw.checked;
    agentSwitchUserTouched = true;
    document.getElementById('agentSwitchLabel').textContent =
      enabled ? 'Send prompt to Agent' : 'Agent off — direct to Isaac Sim';
    document.querySelector('.agent-switch-row').classList.toggle('on', enabled);
    try {
      await fetch(bp + '/agent_toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
    } catch(e) {
    } finally {
      agentSwitchUserTouched = false;
    }
  }

  async function sendHardReset() {
    // Immediately stops any running prompt/agentic plan, homes the arm, and resets every
    // cube to its default staging spot (see /hard_reset). Clears the prompt box so the
    // user isn't looking at a now-abandoned instruction, and re-enables Execute right away
    // rather than waiting on the next /state poll to see busy flip back to false.
    const hardBtn = document.getElementById('btnHardReset');
    hardBtn.disabled = true;
    try {
      await fetch(bp + '/hard_reset', { method: 'POST' });
      document.getElementById('promptInput').value = '';
      document.getElementById('btnExecute').disabled = false;
    } catch(e) {
    } finally {
      hardBtn.disabled = false;
    }
  }

  async function sendRandomize() {
    try {
      await fetch(bp + '/randomize', { method: 'POST' });
    } catch(e) {}
  }
</script>
</body>
</html>
"""

# ─── API ROUTES ───────────────────────────────────────────────────────────────
def _pretty_model(model_id: str) -> str:
    """'nvidia/nemotron-3.5-lightning-30b-a3b' -> 'Nemotron 3.5 Lightning 30B-A3B' (follows the configured model)."""
    words = []
    for w in model_id.split("/")[-1].split("-"):
        if re.fullmatch(r"\d+(\.\d+)?b|a\d+b", w, re.I):
            words.append(w.upper())
        else:
            words.append(w if re.search(r"\d", w) else w.capitalize())
    return re.sub(r"(\d+B) (A\d+B)", r"\1-\2", " ".join(words))


def _nemotron_label(short: bool = False) -> str:
    m = _pretty_model(cfg.NEMOTRON_MODEL or "Nemotron")
    if short:
        m = re.sub(r"-A\d+B$", "", m)                      # '30B-A3B' -> '30B' for the header
    return m if m.lower().startswith("nemotron") else f"Nemotron ({m})"


def _stack_line() -> str:
    """The header subtitle: the stack that is actually configured, not a hard-coded description."""
    return " · ".join(["Isaac Sim 6.0.1", _nemotron_label(short=True), "Cosmos 3 Nano Reasoner"])


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML.replace("__STACK_LINE__", _stack_line()).replace("__NEMOTRON_CHIP__", f"{_nemotron_label()} (NVIDIA API)")


@app.get("/ntt_data_logo.png")
async def ntt_logo():
    path = Path(__file__).parent / "static" / "ntt_data_logo.png"
    if not path.is_file():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})

@app.get("/health")
async def health():
    return {"status": "ok", "busy": sim_state["busy"], "stage": sim_state["stage"],
            "isaac_connected": sim_state["isaac_connected"], "agentic_mode": agent_enabled}

@app.get("/state")
async def get_state():
    return JSONResponse(sim_state)

@app.get("/perception")
async def get_perception():
    """Direct passthrough to Isaac Sim's own /perception (cosmos_integration.md Step 6) —
    same data already embedded in /state's "perception" key at 1 Hz, exposed here too for
    anything that wants the freshest read without waiting on the poll cadence."""
    try:
        req = urllib.request.Request(f"{ISAAC_SIM_URL}/perception", headers={"User-Agent": "Physyk-Bridge"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return JSONResponse(json.loads(resp.read().decode("utf-8")))
    except Exception as e:
        return JSONResponse({"mode": None, "updated": False, "error": f"Isaac Sim unreachable: {e}", "objects": {}}, status_code=200)

@app.post("/execute")
async def execute_route(body: dict):
    instruction = body.get("instruction", "")
    if not instruction:
        return JSONResponse({"error": "No instruction"}, status_code=400)
    result = execute_instruction(instruction)
    return JSONResponse(result)

# Note: there used to be a second, duplicate `@app.post("/reset")` route here — FastAPI
# matches routes in registration order, so this one was always dead/unreachable (the
# `proxy_reset` route defined earlier in the file, near the other proxy routes, is the one
# that actually handled every /reset request). Removed rather than leave a route users would
# reasonably assume was live.

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=cfg.PHYSYK_GUI_PORT)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    push_log("Physyk Physical AI server starting…")
    push_log(f"Connected to Isaac Sim 6.0.1 engine at {cfg.ISAAC_SIM_URL}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
