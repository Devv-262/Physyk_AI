#!/usr/bin/env bash
# ==============================================================================
# 🦾 Physyk AI — Master Unified Runner (run.sh)
# ==============================================================================
# Usage:
#   ./run.sh          -> Restarts Isaac Sim (8100) + Web GUI (7860) ONLY.
#                        Nemotron (8000) and Cosmos (8001), if already running,
#                        are left untouched — this is now safe to run at any time
#                        to bounce the sim without taking the AI stack down with it.
#   ./run.sh --all    -> Restarts Isaac Sim (8100) + Web GUI (7860) + Nemotron-30B (8000).
#                        Cosmos (8001) is still left untouched either way.
#   ./run.sh --status -> Displays real-time health of all services
#   ./run.sh --stop   -> Stops Isaac Sim + GUI + Nemotron (full stop, opt-in only
#                        via this explicit flag). Cosmos is never stopped by this
#                        script — it isn't started by it either.
# ==============================================================================

set -euo pipefail

WS="/opt/dlami/nvme/physyk/workspace"
ISAAC_PYTHON="/opt/dlami/nvme/isaac-sim/python.sh"
VENV_PYTHON="/opt/dlami/nvme/physyk/workspace/Isaac-GR00T/.venv/bin/python3"
LOG_DIR="/opt/dlami/nvme/physyk/logs"
# Sentinel the watchdog (nemotron_watchdog.sh) checks before auto-restarting Nemotron —
# present means "a human explicitly asked for Nemotron to be down", so the watchdog must
# not fight that. Touched only by an explicit `--stop`; removed whenever this script (or
# start_nemotron.sh) is about to actually (re)launch Nemotron.
NEMOTRON_STOPPED_FLAG="/opt/dlami/nvme/physyk/logs/.nemotron_intentionally_stopped"
mkdir -p "$LOG_DIR" /tmp

MODE="${1:-default}"

# ── Function: Stop Services ───────────────────────────────────────────────────
# scope="core" (default) only touches what THIS script is about to restart itself —
# the GUI (7860) and Isaac Sim (8100). It must never kill Nemotron (8000) or Cosmos
# (8001, started separately via cosmos_integration.md's scripts and not managed here
# at all) as a side effect of an unrelated Isaac Sim/GUI bounce — that used to happen
# because this function unconditionally fuser-killed port 8000 and pkilled every
# "vllm.entrypoints.openai" process, so a plain `./run.sh` silently took Nemotron down
# and never brought it back (only `--all`/`--ai` restarts it). scope="all" is the
# explicit, opt-in path that also stops Nemotron — and even then the pkill is scoped
# to the Nemotron-30B model path specifically (matching start_nemotron.sh's existing
# pattern) so Cosmos on the same vllm.entrypoints.openai binary is never collateral.
stop_services() {
    local scope="${1:-core}"
    echo "🔴 Stopping Physyk services (scope=${scope})..."
    fuser -k 7860/tcp 8100/tcp 2>/dev/null || true
    pkill -9 -f "physyk_main_server" 2>/dev/null || true
    pkill -9 -f "isaac_sim_service"   2>/dev/null || true
    if [ "$scope" = "all" ]; then
        fuser -k 8000/tcp 2>/dev/null || true
        pkill -9 -f "vllm.entrypoints.openai.*Nemotron-30B" 2>/dev/null || true
        pkill -9 -f "vllm serve.*Nemotron-30B" 2>/dev/null || true
    fi
    sleep 2
    echo "✅ Services stopped (Nemotron/Cosmos left untouched)."
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
        echo "  ✅ Nemotron-30B Brain (8000): RUNNING -> http://localhost:8000/v1/models"
    else
        echo "  ⚪ Nemotron-30B Brain (8000): OFFLINE (Run './run.sh --all' to start)"
    fi

    # 4. Isaac Sim WebRTC (8210)
    if curl -sf http://localhost:8210/ > /dev/null 2>&1; then
        echo "  ✅ Isaac Sim WebRTC (8210):   RUNNING -> http://localhost:8210"
    else
        echo "  ⚪ Isaac Sim WebRTC (8210):   OFFLINE"
    fi

    echo "══════════════════════════════════════════════════════════════════"
}

if [ "$MODE" = "--stop" ] || [ "$MODE" = "stop" ]; then
    # Explicit full stop, opted into by name — this is the one call site allowed to
    # also take Nemotron down, since the user asked to stop everything, not restart
    # Isaac Sim/GUI. Cosmos still isn't touched (see stop_services comment above).
    # Set the sentinel BEFORE killing so nemotron_watchdog.sh sees "intentional" and
    # doesn't race to auto-restart it the moment the port drops.
    touch "$NEMOTRON_STOPPED_FLAG"
    stop_services all
    exit 0
fi

if [ "$MODE" = "--status" ] || [ "$MODE" = "status" ]; then
    check_status
    exit 0
fi

# ── Launch Sequence ───────────────────────────────────────────────────────────
# Only stop (and later restart) Nemotron when this run is actually going to relaunch
# it — i.e. --all/--ai. A plain ./run.sh restarting Isaac Sim/GUI must leave an
# already-running Nemotron (or Cosmos) alone.
if [ "$MODE" = "--all" ] || [ "$MODE" = "--ai" ] || [ "$MODE" = "all" ]; then
    stop_services all
else
    stop_services core
fi

echo "🚀 [1/3] Starting Isaac Sim Physics Engine (Port 8100)..."
setsid "$ISAAC_PYTHON" "$WS/isaac_sim_service.py" > /tmp/isaac_sim_service.log 2>&1 &
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

echo "🌐 [2/3] Starting Physyk Web GUI (Port 7860)..."
setsid python3 "$WS/physyk_main_server.py" --port 7860 > /tmp/physyk_server.log 2>&1 &
GUI_PID=$!
echo "   PID: $GUI_PID | Log: /tmp/physyk_server.log"

sleep 3

# Optional: Launch Nemotron-30B AI vLLM Server if --all or --ai specified
# Delegates to start_nemotron.sh rather than keeping a second, separately-tunable copy of
# the same vllm invocation here — the two previously drifted (this block hardcoded
# --max-model-len 4096 while start_nemotron.sh and the orchestrator's own code both assumed
# 12288), which silently 400'd every Nemotron call the moment someone restarted via this
# path instead of that script. One launch definition, in start_nemotron.sh, is the fix.
if [ "$MODE" = "--all" ] || [ "$MODE" = "--ai" ] || [ "$MODE" = "all" ]; then
    echo "🧠 [3/3] Starting Nemotron-30B vLLM Inference Brain (Port 8000)..."
    rm -f "$NEMOTRON_STOPPED_FLAG"
    nohup bash "$WS/start_nemotron.sh" > "$LOG_DIR/nemotron_server.log" 2>&1 &
    NEMO_PID=$!
    echo "   PID: $NEMO_PID | Log: $LOG_DIR/nemotron_server.log"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  🟢 PHYSYK STACK IS LIVE!"
echo "══════════════════════════════════════════════════════════════════"
echo "  🖥️  Web GUI & Cameras:  http://localhost:7860"
echo "  🎥  Camera Stream:      http://localhost:8100/camera/stream"
if [ "$MODE" = "--all" ] || [ "$MODE" = "--ai" ] || [ "$MODE" = "all" ]; then
echo "  🧠  Nemotron AI Brain:  http://localhost:8000/v1/models"
fi
echo ""
echo "  📌 Logs:"
echo "     tail -f /tmp/physyk_server.log"
echo "     tail -f /tmp/isaac_sim_service.log"
if [ "$MODE" = "--all" ] || [ "$MODE" = "--ai" ] || [ "$MODE" = "all" ]; then
echo "     tail -f $LOG_DIR/nemotron_server.log"
fi
echo "══════════════════════════════════════════════════════════════════"
