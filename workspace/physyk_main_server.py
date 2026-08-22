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
import json
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

# Real Phase 5 agentic orchestrator (Nemotron decomposition + guardrails + dispatch/verify
# loop) — pure HTTP client of the Nemotron (8000) and Isaac Sim (8100) servers, so it's cheap
# to import here under plain python3. Previously this block also imported/instantiated
# GR00TN17PolicyService() (which loads a real model in-process) and an unused ROS2RGBDBridge;
# neither is used by anything in this file — the real GR00T VLA path lives entirely in its own
# server (port 8300) plus isaac_sim_service.py's HTTP client of it. GR00TN17PolicyService()
# also can't actually load here (the `gr00t` package isn't installed under plain python3,
# only in Isaac-GR00T/.venv) — confirmed it was silently failing and getting swallowed by the
# `except ImportError` below (ModuleNotFoundError is a subclass of ImportError) every time
# this server started, so nemo_agent was never actually usable. Removed both.
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

ISAAC_SIM_URL = "http://localhost:8100"
NEMOTRON_URL = "http://localhost:8000"
COSMOS_URL = "http://localhost:8001"
GROOT_SERVER_URL = "http://localhost:8300"

app = FastAPI(title="Physyk Franka Panda AI Control")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Nemotron/Cosmos full-reasoning pages — /reasoning/nemotron, /reasoning/cosmos. Mounted here
# (same app, same port as everything else in this file) rather than on their own ports, since
# only 7860/8211 are actually forwarded through Brev/VS Code (see BREV_PORT_CONFIG.md) — a
# separate port would work over plain localhost but 404/fail to connect for anyone reaching
# this dashboard remotely, which is exactly what happened with the first version of this.
from reasoning_dashboard import mount_reasoning_routes
mount_reasoning_routes(app)

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
    # Real GR00T VLA per-chunk summary (isaac_sim_service.py's own vla_reasoning field) —
    # None unless a "vla:"-prefixed task has actually run at least one chunk through the
    # model. Not narration: a real summary of the model's own predicted action_chunk tensor.
    "vla_reasoning": None,
    # Structured chunk counter alongside vla_reasoning — was declared here but never
    # actually forwarded from Isaac Sim's /state in the poll loop below, so this field was
    # silently stuck at Python's implicit None (missing key) forever regardless of what
    # Isaac Sim reported. Chunk progress was still visible via vla_reasoning's own text and
    # via `stage`, so nothing downstream broke, but any future UI wanting the structured
    # count (e.g. a progress bar) was reading a permanently-null field. Fixed alongside.
    "vla_chunks_done": None,
    # Live online/offline status for all 4 backing services, independent of whether the
    # "Send to Agent" toggle is on — the toggle only changes dispatch ROUTING (see
    # /agent_toggle's own docstring), it never starts or stops anything, so this panel is
    # meant to reflect real process health regardless of that toggle's position.
    "service_status": {
        "isaac_sim": False,
        "nemotron": False,
        "cosmos": False,
        "groot": False,
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
                    sim_state["vla_reasoning"] = data.get("vla_reasoning")
                    sim_state["vla_chunks_done"] = data.get("vla_chunks_done")
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

def _url_is_healthy(url: str, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Physyk-Bridge"})
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
        sim_state["service_status"]["isaac_sim"] = sim_state["isaac_connected"]
        sim_state["service_status"]["nemotron"] = _url_is_healthy(f"{NEMOTRON_URL}/health")
        sim_state["service_status"]["cosmos"] = _url_is_healthy(f"{COSMOS_URL}/v1/models")
        sim_state["service_status"]["groot"] = _url_is_healthy(f"{GROOT_SERVER_URL}/health")
        time.sleep(5.0)

service_status_thread = threading.Thread(target=poll_service_status, daemon=True)
service_status_thread.start()

def _forward_direct_to_isaac(instruction: str):
    """Fire-and-forget single instruction straight to Isaac Sim's /execute — the original
    dispatch path, kept as-is for plain control commands (reset/home/randomize/vla:) that
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
    if inst.startswith("vla:") or inst.startswith("vla "):
        return True
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
                cv2.putText(img, "ISAAC SIM 6.0.1 - CONNECTING TO GPU RENDERER...", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
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

@app.get("/camera/front.stream")
async def proxy_front_stream():
    return await proxy_mjpeg_stream("/camera/front.stream")

@app.get("/camera/side.stream")
async def proxy_side_stream():
    return await proxy_mjpeg_stream("/camera/side.stream")

@app.get("/camera/top.stream")
async def proxy_top_stream():
    return await proxy_mjpeg_stream("/camera/top.stream")

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
            {"id": "front", "name": "Front Operator View", "stream": "/camera/front.stream"},
            {"id": "side", "name": "Side Profile & Height Clearance", "stream": "/camera/side.stream"},
            {"id": "top", "name": "Top-Down Bird's Eye Planar", "stream": "/camera/top.stream"},
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
      2. Isaac Sim's own physics loop, which may be mid-motion on a pick_place or VLA task
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
    --accent: #5b9bff;
    --accent-glow: rgba(91, 155, 255, 0.22);
    --accent2: #8b5cf6;
    --cyan: #22d3ee;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --text: #eef1f7;
    --muted: #7e879a;
    --muted-2: #59626f;
    --r: 14px;
    --r-sm: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scrollbar-color: #2a3142 transparent; }
  body {
    background:
      radial-gradient(1200px 700px at 12% -10%, rgba(91,155,255,0.08), transparent 60%),
      radial-gradient(900px 600px at 100% 0%, rgba(139,92,246,0.06), transparent 55%),
      var(--bg);
    color: var(--text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh; overflow-x: hidden;
  }
  ::-webkit-scrollbar { width: 9px; height: 9px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #262e3d; border-radius: 999px; }
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
    width: 38px; height: 38px; background: linear-gradient(135deg,#5b9bff,#8b5cf6); border-radius: 11px;
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
  .status-pill.feature { color: var(--cyan); border-color: rgba(34,211,238,0.28); background: rgba(34,211,238,0.08); }
  .status-pill.feature .pill-icon { filter: drop-shadow(0 0 4px rgba(34,211,238,0.6)); }
  .status-pill.mode-agentic { color: var(--accent2); border-color: rgba(139,92,246,0.28); background: rgba(139,92,246,0.08); }
  .status-pill.mode-direct { color: var(--amber); border-color: rgba(245,158,11,0.28); background: rgba(245,158,11,0.08); }
  .status-pill.mode-shadow, .status-pill.mode-stage { color: var(--cyan); border-color: rgba(34,211,238,0.28); background: rgba(34,211,238,0.08); }
  .status-pill.mode-vision { color: var(--green); border-color: rgba(74,222,128,0.28); background: rgba(74,222,128,0.08); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  .dot.busy { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 0.6s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* 4-service status strip — real process health for Isaac Sim / Nemotron / Cosmos / GR00T,
     independent of the agent toggle (a service can be ONLINE while the toggle routes
     around it). See poll_service_status() in the Python backend. */
  .svc-strip { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .svc-pill {
    display: flex; align-items: center; gap: 6px; padding: 5px 11px;
    background: var(--card); border: 1px solid var(--border); border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
  }
  .svc-pill .svc-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .svc-pill.svc-online { color: var(--green); border-color: rgba(74,222,128,0.28); background: rgba(74,222,128,0.08); }
  .svc-pill.svc-online .svc-dot { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .svc-pill.svc-offline { color: var(--muted); border-color: var(--border); background: rgba(148,163,184,0.06); }
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
    background: var(--card-2); color: #dfe4ee; border: 1px solid var(--border); border-radius: 7px;
    padding: 5px 9px; font-size: 0.74rem; outline: none; cursor: pointer; font-weight: 600; font-family: inherit;
  }
  .vp-select:hover { border-color: #333d50; }
  .vp-badge { font-size: 0.68rem; padding: 3px 10px; border-radius: 999px; background: rgba(91,155,255,0.14); color: var(--accent); border: 1px solid rgba(91,155,255,0.28); font-weight: 700; }
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
  .telem-card:hover { border-color: #2d3648; transform: translateY(-1px); }
  .telem-label {
    font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.9px; font-weight: 700;
    margin-bottom: 7px; display: flex; align-items: center; justify-content: space-between; gap: 6px;
  }
  .telem-tag { font-size: 0.56rem; padding: 1px 6px; border-radius: 999px; font-weight: 800; letter-spacing: 0.3px; }
  .telem-tag.live { background: rgba(34,197,94,0.14); color: var(--green); }
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
  .switch input:checked + .switch-track { background: rgba(139,92,246,0.22); border-color: var(--accent2); }
  .switch input:checked + .switch-track .switch-thumb { transform: translateX(18px); background: var(--accent2); }
  .agent-switch-label { font-size: 0.78rem; font-weight: 600; color: var(--muted); }
  .agent-switch-row.on .agent-switch-label { color: var(--text); }
  .btn {
    flex: 1; padding: 11px 16px; border: none; border-radius: 9px; cursor: pointer;
    font-size: 0.83rem; font-weight: 700; font-family: inherit; transition: all 0.15s ease;
  }
  .btn-primary { background: linear-gradient(135deg,#5b9bff,#8b5cf6); color: #fff; box-shadow: 0 4px 14px var(--accent-glow); }
  .btn-primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 7px 22px rgba(91,155,255,0.38); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
  .btn-secondary { background: var(--card-2); border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { background: #232b39; border-color: #333d50; }
  .btn-danger { background: linear-gradient(135deg,#ef4444,#b91c1c); color: #fff; box-shadow: 0 4px 14px rgba(239,68,68,0.35); }
  .btn-danger:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 7px 22px rgba(239,68,68,0.5); }
  .btn-danger:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* ── QUICK TASKS ── */
  .quick-cmds { padding: 18px 20px; border-bottom: 1px solid var(--border-soft); }
  .cmd-grid { display: grid; grid-template-columns: 1fr; gap: 7px; }
  .cmd-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 9px;
    padding: 10px 14px; font-size: 0.79rem; cursor: pointer; display: flex; align-items: center; gap: 10px;
    transition: all 0.15s ease; font-weight: 500;
  }
  .cmd-card:hover { border-color: var(--accent); background: rgba(91,155,255,0.07); transform: translateX(2px); }
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
  .log-msg.SUCCESS { color: #4ade80; }
  .log-msg.WARN { color: #fbbf24; }
  .log-msg.ERROR { color: #f87171; }

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
  .badge-RUNNING, .plan-status-PLANNING.active, .plan-status-RUNNING { background: rgba(91,155,255,0.16); color: var(--accent); }
  .plan-step.RETRYING, .plan-step.REPLANNING { border-color: #fbbf24; }
  .badge-RETRYING, .badge-REPLANNING, .plan-status-PARTIAL, .plan-status-VERIFYING { background: rgba(251,191,36,0.16); color: #fbbf24; }
  .plan-step.DONE { border-color: rgba(74,222,128,0.4); }
  .badge-DONE, .plan-status-SUCCESS { background: rgba(74,222,128,0.16); color: #4ade80; }
  .plan-step.FAILED, .plan-step.REJECTED { border-color: rgba(248,113,113,0.4); }
  .badge-FAILED, .badge-REJECTED, .plan-status-FAILED, .plan-status-REJECTED, .plan-status-GOAL_NOT_HELD { background: rgba(248,113,113,0.16); color: #f87171; }
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
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg class="icon-lg" viewBox="0 0 24 24"><path d="M4 20h4l2-7-3-6"/><circle cx="7" cy="6" r="1.6"/><path d="M10 13l6-2 3 3-2 6h-4"/><circle cx="19" cy="12" r="1.4"/></svg>
    </div>
    <div class="logo-text">
      <h1>Physyk AI</h1>
      <p>NVIDIA Isaac Sim 6.0.1 • RTX PRO 6000 Blackwell (96 GB) • Franka Panda 7-DOF</p>
    </div>
  </div>
  <div class="header-badges">
    <div class="status-pill feature"><span class="pill-icon"><svg class="icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M14.5 9.5l-2 5-3-1 2-5z"/></svg></span><span>RMPFlow · Collision-Aware</span></div>
    <div class="status-pill" id="modePill" title="Whether instructions go through the Nemotron agentic orchestrator (decompose/guardrail/verify/retry) or straight to Isaac Sim as typed.">
      <span class="pill-icon" id="modePillIcon"></span><span id="modePillText">—</span>
    </div>
    <div class="status-pill" id="perceptionPill" title="Cosmos-Reason2-2B vision perception mode. Stage/shadow: ground-truth pose drives the robot, Cosmos runs alongside and only logs a delta. Vision: Cosmos's own estimate drives the robot (with automatic fallback to ground truth on failure).">
      <span class="pill-icon"><svg class="icon-sm" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="2.6"/></svg></span><span id="perceptionPillText">PERCEPTION —</span>
    </div>
    <div class="status-pill">
      <div class="dot" id="statusDot"></div>
      <span id="statusText">ISAAC SIM ONLINE</span>
    </div>
  </div>
  <div class="svc-strip" title="Real process health for all 4 backing services. A service being OFFLINE here means it isn't running at all — separate from the 'Send to Agent' toggle above, which only changes routing and never stops anything.">
    <div class="svc-pill svc-offline" id="svc-isaac_sim"><span class="svc-dot"></span><span class="svc-label">Isaac Sim</span><span class="svc-state">—</span></div>
    <div class="svc-pill svc-offline" id="svc-nemotron"><span class="svc-dot"></span><span class="svc-label">Nemotron</span><span class="svc-state">—</span></div>
    <div class="svc-pill svc-offline" id="svc-cosmos"><span class="svc-dot"></span><span class="svc-label">Cosmos</span><span class="svc-state">—</span></div>
    <div class="svc-pill svc-offline" id="svc-groot"><span class="svc-dot"></span><span class="svc-label">GR00T VLA</span><span class="svc-state">—</span></div>
    <a href="/reasoning/nemotron" target="_blank" class="status-pill" style="text-decoration:none; cursor:pointer;" title="Full, un-truncated Nemotron reasoning log">🧠 Nemotron Log</a>
    <a href="/reasoning/cosmos" target="_blank" class="status-pill" style="text-decoration:none; cursor:pointer;" title="Full, un-truncated Cosmos-Reason2 perception log">👁️ Cosmos Log</a>
  </div>
</header>

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
            <option value="front">Front View (Operator Facing)</option>
            <option value="side">Side Profile (Depth &amp; Height)</option>
            <option value="top">Top-Down (Bird's Eye Planar)</option>
            <option value="wrist">Wrist Camera (Gripper Close-up)</option>
          </select>
          <span class="vp-badge" id="feedStatus">HTTP Stream</span>
        </div>
        <span class="fps-readout"><span class="live-tag">● LIVE</span><span id="fpsBadge">— FPS</span></span>
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
    </div>

    <!-- Quick Tasks -->
    <div class="quick-cmds">
      <div class="section-title">Quick Action Tasks</div>
      <div class="cmd-grid">
        <div class="cmd-card" onclick="sendRandomize()" style="border-color: rgba(245,158,11,0.35);" title="Places all 3 cubes at genuinely random positions (np.random, not a fixed list) anywhere within the arm's reach, clear of the tray.">
          <span class="cmd-swatch" style="background:#f59e0b;"></span>
          <span><svg class="icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="15.5" cy="8.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="8.5" cy="15.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/></svg> Randomize Cube Positions</span>
        </div>
        <div class="cmd-card" onclick="runQuick('Return robot arm to home configuration')" title="Moves only the robot arm — cube positions are left exactly where they are.">
          <span class="cmd-swatch" style="background:#7e879a;"></span>
          <span>Return to Home Pose (arm only)</span>
        </div>
        <div class="cmd-card" onclick="sendReset()" style="border-color: rgba(34,211,238,0.3);">
          <span class="cmd-swatch" style="background:#22d3ee;"></span>
          <span>Full Reset — Cubes to Fixed Spots + Home</span>
        </div>
        <div class="cmd-card" onclick="runQuick('vla: pick up the red cube and place it in the tray')" style="border-color: rgba(139,92,246,0.35);" title="Real closed-loop GR00T-N1.7 inference (camera + language -> action) driving the arm directly, instead of the deterministic PickPlaceController. Not fine-tuned on this scene, so it may not complete the task &#8212; this exercises the real model loop, not scripted motion.">
          <span class="cmd-swatch" style="background:#8b5cf6;"></span>
          <span><svg class="icon" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9z"/><path d="M4 9h2M4 15h2M18 9h2M18 15h2M9 4v2M15 4v2M9 18v2M15 18v2"/></svg> Try Real GR00T VLA (experimental)</span>
        </div>
      </div>
      <p style="font-size:0.72rem; color:var(--muted); margin-top:10px; line-height:1.5;">
        Type your own instructions above — e.g. "pick up the green cube and place it in the tray".
        There's no per-color shortcut button anymore; the model has to actually find the cube
        you asked for, wherever it randomly ended up.
      </p>
    </div>

    <!-- Real Agentic Plan (Nemotron decompose -> guardrail -> dispatch -> verify -> replan) -->
    <div class="plan-panel">
      <div class="section-title"><svg class="icon" viewBox="0 0 24 24"><circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="13" r="2.4"/><circle cx="6" cy="19" r="2.4"/><circle cx="18" cy="19" r="2.4"/><path d="M7.7 7.5L10.5 11.5M16.3 7.5L13.5 11.5M10.7 14.8L7.3 17.7M13.3 14.8L16.7 17.7"/></svg> Agent Plan</div>
      <div id="planBox">
        <div class="plan-empty">No plan running — type an instruction above.</div>
      </div>
      <div id="vlaReasoningBox" class="plan-step-detail" style="display:none; margin-top:10px; padding:8px; border-left:2px solid #a78bfa; background:rgba(167,139,250,0.06);"></div>
    </div>

    <!-- Cosmos-Reason2 shadow-mode perception (cosmos_integration.md Step 6) -->
    <div class="plan-panel">
      <div class="section-title"><svg class="icon" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="2.6"/></svg> Vision Perception (Cosmos-Reason2)</div>
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
        <div class="api-row"><span>Front Operator View Stream</span><a data-path="/camera/front.stream" target="_blank">Open <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>Side Profile View Stream</span><a data-path="/camera/side.stream" target="_blank">Open <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
        <div class="api-row"><span>Top-Down Bird's Eye Stream</span><a data-path="/camera/top.stream" target="_blank">Open <svg class="icon-sm" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4L10 14M9 5H5a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg></a></div>
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
  <span class="stack-chip">Isaac Sim 6.0.1</span>
  <span class="stack-chip">PhysX (TGS)</span>
  <span class="stack-chip">RMPFlow</span>
  <span class="stack-chip">PickPlaceController</span>
  <span class="stack-chip">Franka Panda 7-DOF</span>
</footer>

<script>
  const bp = window.location.pathname.replace(/\/$/, '');
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
    else if (verified) { color = '#4ade80'; label = 'VERIFIED'; }
    else { color = '#fbbf24'; label = 'VISUAL CHECK DISAGREES'; }
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
      return `<div class="plan-step-detail" style="color:var(--muted); margin-top:4px;">👁 COSMOS: ${a.error}</div>`;
    }
    const feasColor = a.feasibility === 'high' ? '#4ade80' : (a.feasibility === 'medium' ? '#fbbf24' : (a.feasibility === 'low' ? '#f87171' : 'var(--muted)'));
    const tv = a.target_visible === true ? '✓' : (a.target_visible === false ? '✗' : '?');
    const dv = a.destination_visible === true ? '✓' : (a.destination_visible === false ? '✗' : (a.destination_visible === null ? '—' : '?'));
    let html = `<div class="plan-step-detail" style="margin-top:4px;">
      <span style="font-weight:700; letter-spacing:0.4px; color:#60a5fa;">👁 COSMOS REASON 2</span>
      &nbsp;target ${tv} &nbsp;dest ${dv}
      &nbsp;feasibility <span style="color:${feasColor}; font-weight:700;">${(a.feasibility || '?').toUpperCase()}</span>`;
    if (a.approach) html += `&nbsp;· approach: ${a.approach}`;
    if (a.obstacle && a.obstacle.toLowerCase() !== 'none') html += `&nbsp;· obstacle: ${a.obstacle}`;
    if (a.reasoning) html += `&nbsp;· <a href="${COSMOS_LOG_URL}" target="_blank" style="color:#34d399;">full reasoning →</a>`;
    html += '</div>';
    return html;
  }

  // Nemotron's own diagnosis text when it decided retry/replan on a real failure — one-line
  // summary here (the decision it made), full chain-of-thought on its own dashboard.
  function renderPlannerReasoning(step) {
    if (!step.planner_reasoning) return '';
    return `<div class="plan-step-detail" style="color:var(--muted); margin-top:4px;">
      🧠 Nemotron diagnosed this failure and decided how to proceed —
      <a href="${NEMOTRON_LOG_URL}" target="_blank" style="color:#60a5fa;">full reasoning →</a>
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
      html += `<div class="plan-step-detail" style="color:var(--muted); margin:6px 0 10px; padding:8px; border-left:2px solid #60a5fa; background:rgba(96,165,250,0.06);">
        🧠 <span style="font-weight:700; color:#60a5fa;">NEMOTRON PLANNING</span> — decomposed into ${stepCount} step${stepCount === 1 ? '' : 's'}.
        <a href="${NEMOTRON_LOG_URL}" target="_blank" style="color:#60a5fa;">full reasoning →</a>
      </div>`;
    }
    html += '<div class="plan-steps">';
    for (const step of (plan.steps || [])) {
      const status = step.status || 'PENDING';
      html += `<div class="plan-step ${status}">
        <div class="plan-step-idx">${step.index}</div>
        <div class="plan-step-body">
          <div class="plan-step-text">${step.text || ''}</div>
          ${step.detail ? `<div class="plan-step-detail">${step.detail}</div>` : ''}
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
  const PERC_LABELS = { red_cube: 'Red Cube', blue_cube: 'Blue Cube', green_cube: 'Green Cube', target_tray: 'Target Tray' };
  let lastPerceptionJson = null;
  function renderPerception(perc) {
    const box = document.getElementById('perceptionBox');
    const pill = document.getElementById('perceptionPill');
    const pillText = document.getElementById('perceptionPillText');
    const json = JSON.stringify(perc);
    if (json === lastPerceptionJson) return;
    lastPerceptionJson = json;

    const mode = perc ? perc.mode : null;
    if (mode) {
      pill.className = 'status-pill mode-' + mode;
      pillText.textContent = 'PERCEPTION: ' + mode.toUpperCase();
    } else {
      pill.className = 'status-pill';
      pillText.textContent = 'PERCEPTION —';
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
        deltaColor = delta < 20 ? '#4ade80' : (delta < 80 ? '#fbbf24' : '#f87171');
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
      html += `<div class="plan-step-detail" style="margin-top:8px;">Cosmos grounding latency: ${perc.latency_ms.toFixed(0)} ms</div>`;
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
          modeText.textContent = 'Agentic (Nemotron)';
        } else {
          modePill.className = 'status-pill mode-direct';
          modeIcon.innerHTML = '<svg class="icon-sm" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg>';
          modeText.textContent = 'Direct (no agent)';
        }

        // 4-service status panel — reflects real process health (poll_service_status in
        // Python), independent of the agent toggle: a service can be RUNNING while the
        // toggle is OFF (it's just not being dispatched to), and OFFLINE services show
        // clearly rather than the panel silently omitting them.
        if (data.service_status) {
          const svcLabels = {isaac_sim: 'Isaac Sim', nemotron: 'Nemotron', cosmos: 'Cosmos', groot: 'GR00T VLA'};
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

        // Real GR00T VLA per-chunk summary (only present once a "vla:" task has run at
        // least one chunk) — rendered into the same plan box area used for the agent plan,
        // since VLA tasks bypass the Nemotron orchestrator entirely (opt-in "vla:" prefix).
        const vlaBox = document.getElementById('vlaReasoningBox');
        if (vlaBox) {
          if (data.vla_reasoning) {
            vlaBox.style.display = '';
            vlaBox.innerHTML = `<span style="font-weight:700; color:#a78bfa;">🤖 GR00T VLA</span> — ${data.vla_reasoning}`;
          } else {
            vlaBox.style.display = 'none';
          }
        }

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
        if (data.busy) {
            dot.className = 'dot busy';
            document.getElementById('statusText').textContent = 'SIMULATION BUSY';
            btn.disabled = true;
        } else {
            dot.className = 'dot';
            document.getElementById('statusText').textContent = 'SYSTEM IDLE';
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
      if (currentFeed === 'front') endpoint = '/camera/front.stream';
      else if (currentFeed === 'side') endpoint = '/camera/side.stream';
      else if (currentFeed === 'top') endpoint = '/camera/top.stream';
      else if (currentFeed === 'wrist') endpoint = '/camera/wrist.stream';

      liveFeed.src = bp + endpoint;
      liveFeed.onload = () => {
          const names = {
            scene: 'Overview', front: 'Front View', side: 'Side Profile', top: 'Top-Down', wrist: 'Wrist'
          };
          feedStatus.innerHTML = '<span style="color:#4ADE80;">●</span> Live (' + (names[currentFeed] || currentFeed) + ')';
          const overlay = document.getElementById('streamLoadingOverlay');
          if (overlay) overlay.style.display = 'none';
      };
      liveFeed.onerror = () => {
          feedStatus.innerHTML = '<span style="color:#F87171;">●</span> Connecting...';
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
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

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
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    push_log("Physyk Physical AI server starting…")
    push_log("Connected to Isaac Sim 6.0.1 engine on port 8100")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
