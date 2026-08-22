#!/usr/bin/env bash
# ==============================================================================
# 🦾 Physyk AI — Master Unified Runner (run.sh)
# ==============================================================================
# Usage:
#   ./run.sh          -> Starts ALL 4: Isaac Sim (8100) + Web GUI (7860) +
#                         Nemotron-30B (8000) + Cosmos-Reason2 (8001) + GR00T VLA (8300)
#                         [Full stack, recommended]
#   ./run.sh --no-agent -> Starts Isaac Sim (8100) + Web GUI (7860) + GR00T VLA (8300)
#                          WITHOUT Nemotron or Cosmos — every instruction goes straight
#                          to Isaac Sim's /execute exactly as typed, no decompose/
#                          guardrail/verify/retry loop, no perception grounding. GR00T VLA
#                          is still started since "vla:"-prefixed instructions bypass the
#                          orchestrator entirely regardless of this mode — testing the raw
#                          PickPlaceController / VLA paths in isolation, without the
#                          heavier reasoning models' GPU footprint.
#   ./run.sh --status -> Displays real-time health of all 5 services
#   ./run.sh --stop   -> Stops all background Physyk processes (all 5 services)
# ==============================================================================
#
# NOTE on the live "Send to Agent" toggle (in the Web GUI, distinct from --no-agent
# above): flipping it OFF does NOT stop Nemotron or any other service — it only changes
# physyk_main_server.py's dispatch routing so every instruction bypasses the orchestrator
# and goes straight to Isaac Sim's PickPlaceController, exactly like --no-agent's routing,
# but reversible live without restarting anything. See physyk_main_server.py's
# /agent_toggle endpoint and its own docstring for the full behavior.

set -euo pipefail

WS="/opt/dlami/nvme/physyk/workspace"
ISAAC_PYTHON="/opt/dlami/nvme/isaac-sim/python.sh"
VENV_PYTHON="/opt/dlami/nvme/physyk/workspace/.venv-nemotron/bin/python3"
LOG_DIR="/opt/dlami/nvme/physyk/logs"
mkdir -p "$LOG_DIR" /tmp

MODE="${1:-default}"
NO_AGENT=false
if [ "$MODE" = "--no-agent" ] || [ "$MODE" = "no-agent" ]; then
    NO_AGENT=true
fi

# ── Function: Stop Services ───────────────────────────────────────────────────
# Stops everything run.sh itself can start: Isaac Sim, the web GUI, Nemotron, Cosmos, and
# GR00T. Each pattern is scoped as specifically as possible (matched by model dir / script
# name, not a generic "vllm.entrypoints" substring) — a blanket pkill on that substring
# previously killed whichever of Nemotron/Cosmos WASN'T the intended target on every
# restart, a real outage confirmed live this session. Scoping fixes that in both directions.
stop_services() {
    echo "🔴 Stopping all Physyk services..."
    pkill -f physyk_main_server 2>/dev/null || true
    pkill -f isaac_sim_service   2>/dev/null || true
    pkill -f groot_policy_service 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.*Nemotron-30B" 2>/dev/null || true
    pkill -f "vllm serve.*Nemotron-30B" 2>/dev/null || true
    pkill -f "vllm serve.*Cosmos-Reason2" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.*Cosmos-Reason2" 2>/dev/null || true
    # Prevent the Nemotron watchdog (if running) from immediately restarting what we just
    # intentionally stopped — see nemotron_watchdog.sh's own flag-check logic.
    touch "$LOG_DIR/.nemotron_intentionally_stopped" 2>/dev/null || true
    sleep 2
    echo "✅ All services stopped."
}

# ── Function: Status Check ────────────────────────────────────────────────────
check_status() {
    echo "══════════════════════════════════════════════════════════════════"
    echo "  🦾 Physyk System Health & Service Status"
    echo "══════════════════════════════════════════════════════════════════"

    # 1. Physyk Web GUI (7860)
    if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
        echo "  ✅ Physyk Web GUI (7860):     RUNNING -> http://localhost:7860"
    else
        echo "  ❌ Physyk Web GUI (7860):     DOWN"
    fi

    # 2. Isaac Sim Service (8100)
    if curl -sf http://localhost:8100/health > /dev/null 2>&1; then
        echo "  ✅ Isaac Sim Engine (8100):   RUNNING -> http://localhost:8100/camera/stream"
    else
        echo "  ❌ Isaac Sim Engine (8100):   DOWN"
    fi

    # 3. Nemotron-30B AI API (8000)
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✅ Nemotron-30B Brain (8000): RUNNING -> http://localhost:7860/reasoning/nemotron"
    else
        echo "  ⚪ Nemotron-30B Brain (8000): OFFLINE (Run './run.sh' without --no-agent to start)"
    fi

    # 4. Cosmos-Reason2 Perception (8001)
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "  ✅ Cosmos-Reason2 (8001):     RUNNING -> http://localhost:7860/reasoning/cosmos"
    else
        echo "  ⚪ Cosmos-Reason2 (8001):     OFFLINE (Run './run.sh' without --no-agent to start)"
    fi

    # 5. GR00T-N1.7 VLA Policy (8300)
    if curl -sf http://localhost:8300/health > /dev/null 2>&1; then
        echo "  ✅ GR00T-N1.7 VLA (8300):     RUNNING -> http://localhost:8300"
    else
        echo "  ⚪ GR00T-N1.7 VLA (8300):     OFFLINE"
    fi

    # 6. Isaac Sim WebRTC (8210)
    if curl -sf http://localhost:8210/ > /dev/null 2>&1; then
        echo "  ✅ Isaac Sim WebRTC (8210):   RUNNING -> http://localhost:8210"
    else
        echo "  ⚪ Isaac Sim WebRTC (8210):   OFFLINE"
    fi

    echo "══════════════════════════════════════════════════════════════════"
}

if [ "$MODE" = "--stop" ] || [ "$MODE" = "stop" ]; then
    stop_services
    exit 0
fi

if [ "$MODE" = "--status" ] || [ "$MODE" = "status" ]; then
    check_status
    exit 0
fi

# ── Launch Sequence ───────────────────────────────────────────────────────────
stop_services
# stop_services sets the watchdog's "intentionally stopped" flag — clear it again now that
# we're about to (re)start everything on purpose, or the watchdog will refuse to touch
# Nemotron even if it later goes down unexpectedly during this run.
rm -f "$LOG_DIR/.nemotron_intentionally_stopped" 2>/dev/null || true

echo "🚀 [1/5] Starting Isaac Sim Physics Engine (Port 8100)..."
nohup "$ISAAC_PYTHON" "$WS/isaac_sim_service.py" > /tmp/isaac_sim_service.log 2>&1 &
ISAAC_PID=$!
echo "   PID: $ISAAC_PID | Log: /tmp/isaac_sim_service.log"

echo "⏳ Waiting for Isaac Sim to initialize rendering & physics..."
READY=0
for i in $(seq 1 90); do
    sleep 1
    if curl -sf http://localhost:8100/health > /dev/null 2>&1; then
        echo "   ✅ Isaac Sim READY in ${i}s"
        READY=1
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "   ... still warming up (${i}s)"
    fi
done

if [ $READY -eq 0 ]; then
    echo "⚠️ Isaac Sim is taking longer than usual to bind port 8100. Check '/tmp/isaac_sim_service.log'."
fi

echo "🌐 [2/5] Starting Physyk Web GUI (Port 7860)..."
if [ "$NO_AGENT" = true ]; then
    echo "   Mode: NON-AGENTIC — instructions go straight to Isaac Sim, no Nemotron orchestrator."
    PHYSYK_AGENTIC=0 nohup python3 "$WS/physyk_main_server.py" --port 7860 > /tmp/physyk_server.log 2>&1 &
else
    nohup python3 "$WS/physyk_main_server.py" --port 7860 > /tmp/physyk_server.log 2>&1 &
fi
GUI_PID=$!
echo "   PID: $GUI_PID | Log: /tmp/physyk_server.log"

sleep 2

# Nemotron and Cosmos are skipped entirely in --no-agent mode (nothing would call them —
# the orchestrator that uses Nemotron is bypassed, and Cosmos grounding only runs as part
# of that same orchestrated path). GR00T VLA is NOT gated by this — "vla:"-prefixed
# instructions bypass the orchestrator regardless of agent mode (see isaac_sim_service.py's
# own control-command handling), so VLA testing works the same in either mode.
if [ "$NO_AGENT" = false ]; then
    echo "🧠 [3/5] Starting Nemotron-30B vLLM Inference Brain (Port 8000)..."
    nohup "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "/opt/dlami/nvme/physyk/models/Nemotron-30B-FP8" \
      --served-model-name nemotron \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype auto \
      --max-model-len 12288 \
      --gpu-memory-utilization 0.48 \
      --trust-remote-code \
      --no-enable-log-requests > "$LOG_DIR/nemotron_server.log" 2>&1 &
    NEMO_PID=$!
    echo "   PID: $NEMO_PID | Log: $LOG_DIR/nemotron_server.log"

    echo "⏳ Waiting for Nemotron to finish loading (this can take ~2-3 minutes)..."
    NEMO_READY=0
    for i in $(seq 1 180); do
        sleep 1
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ Nemotron READY in ${i}s"
            NEMO_READY=1
            break
        fi
        if [ $((i % 20)) -eq 0 ]; then
            echo "   ... still loading (${i}s)"
        fi
    done
    if [ $NEMO_READY -eq 0 ]; then
        echo "⚠️ Nemotron is taking longer than usual. Check '$LOG_DIR/nemotron_server.log'."
    fi

    # Started only AFTER Nemotron confirms healthy, not in parallel — starting both at once
    # let Cosmos's vLLM engine race Nemotron's own memory ramp-up and fail with "No available
    # memory for the cache blocks" (confirmed live this session). Sequencing avoids that.
    echo "👁️  [4/5] Starting Cosmos-Reason2 Perception Server (Port 8001)..."
    nohup bash "$WS/scripts/01_serve_cosmos.sh" > "$LOG_DIR/cosmos_server.log" 2>&1 &
    COSMOS_PID=$!
    echo "   PID: $COSMOS_PID | Log: $LOG_DIR/cosmos_server.log"

    echo "⏳ Waiting for Cosmos to finish loading..."
    COSMOS_READY=0
    for i in $(seq 1 120); do
        sleep 1
        if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
            echo "   ✅ Cosmos READY in ${i}s"
            COSMOS_READY=1
            break
        fi
        if [ $((i % 20)) -eq 0 ]; then
            echo "   ... still loading (${i}s)"
        fi
    done
    if [ $COSMOS_READY -eq 0 ]; then
        echo "⚠️ Cosmos is taking longer than usual (or failed — check for a GPU memory error). Check '$LOG_DIR/cosmos_server.log'."
    fi
fi

echo "🦾 [5/5] Starting GR00T-N1.7 VLA Policy Server (Port 8300)..."
nohup bash "$WS/start_groot.sh" > "$LOG_DIR/groot_server.log" 2>&1 &
GROOT_PID=$!
echo "   PID: $GROOT_PID | Log: $LOG_DIR/groot_server.log"

echo "⏳ Waiting for GR00T to finish loading..."
GROOT_READY=0
for i in $(seq 1 90); do
    sleep 1
    if curl -sf http://localhost:8300/health > /dev/null 2>&1; then
        echo "   ✅ GR00T READY in ${i}s"
        GROOT_READY=1
        break
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "   ... still loading (${i}s)"
    fi
done
if [ $GROOT_READY -eq 0 ]; then
    echo "⚠️ GR00T is taking longer than usual. Check '$LOG_DIR/groot_server.log'."
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  🟢 PHYSYK STACK IS LIVE!"
echo "══════════════════════════════════════════════════════════════════"
echo "  🖥️  Web GUI & Cameras:      http://localhost:7860"
echo "  🎥  Camera Stream:          http://localhost:8100/camera/stream"
echo "  🦾  GR00T VLA Policy:       http://localhost:8300"
if [ "$NO_AGENT" = true ]; then
echo "  ⚪  Mode:                    NON-AGENTIC (direct dispatch, no Nemotron/Cosmos)"
else
echo "  🧠  Mode:                    AGENTIC (Nemotron decompose/guardrail/verify/retry)"
echo "  🧠  Nemotron reasoning log: http://localhost:7860/reasoning/nemotron"
echo "  👁️   Cosmos reasoning log:   http://localhost:7860/reasoning/cosmos"
fi
echo ""
echo "  📌 Live service status any time:  ./run.sh --status"
echo "  📌 Logs:"
echo "     tail -f /tmp/physyk_server.log"
echo "     tail -f /tmp/isaac_sim_service.log"
echo "     tail -f $LOG_DIR/groot_server.log"
if [ "$NO_AGENT" = false ]; then
echo "     tail -f $LOG_DIR/nemotron_server.log"
echo "     tail -f $LOG_DIR/cosmos_server.log"
fi
echo "══════════════════════════════════════════════════════════════════"
