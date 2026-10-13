#!/usr/bin/env bash
# ==============================================================================
# Physyk AI — local runner: starts the BACKEND + GUI only.
#   Backend+GUI: .venv python, http://localhost:7860
#   Isaac Sim  : start it yourself first — ./start_sim_docker.sh (this script does not launch it)
#   Cosmos     : ./start_cosmos_docker.sh (optional; image checks only)
#   Nemotron   : NVIDIA hosted API (keys in .env — see .env.example)
#
#   ./run.sh              (re)start the backend + GUI
#   ./run.sh --no-agent   same, but every instruction goes straight to Isaac Sim (no LLM calls)
#   ./run.sh --status     health of everything
#   ./run.sh --stop       stop the backend (containers: Ctrl+C the sim terminal / docker stop physyk-cosmos)
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$ROOT/workspace"
LOG_DIR="$ROOT/logs"
PY="$ROOT/.venv/bin/python"
MODE="${1:-default}"
mkdir -p "$LOG_DIR"

[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

healthy() { curl -sf -m 3 "$1" >/dev/null 2>&1; }

stop_backend() { pkill -f "workspace/physyk_main_server.py" 2>/dev/null || true; }
stop_all() {
    stop_backend
    echo "Backend stopped. (Sim: Ctrl+C its terminal or 'docker rm -f physyk-isaac-sim'; Cosmos: 'docker stop physyk-cosmos'.)"
}

check_status() {
    healthy http://localhost:8100/health && echo "  OK   Isaac Sim  (8100)" || echo "  DOWN Isaac Sim  (8100)"
    healthy http://localhost:7860/health && echo "  OK   Backend/GUI (7860) -> http://localhost:7860" || echo "  DOWN Backend/GUI (7860)"
    if [ -n "${NVIDIA_API_KEY:-${NEMOTRON_API_KEY:-}}" ]; then
        echo "  key  NVIDIA API key set (models are called on demand; see GUI status pills)"
    else
        echo "  WARN no NVIDIA_API_KEY in .env — Nemotron/Cosmos calls will fail (Nemotron falls back to single-step dispatch)"
    fi
}

case "$MODE" in
    --stop|stop)     stop_all; exit 0 ;;
    --status|status) check_status; exit 0 ;;
esac

[ -x "$PY" ] || { echo "Missing $PY — create it: python3 -m venv .venv && .venv/bin/pip install fastapi 'uvicorn[standard]' httpx opencv-python-headless numpy pillow"; exit 1; }

if healthy http://localhost:8100/health; then
    echo "Isaac Sim is up on :8100."
else
    echo "WARNING: Isaac Sim is not running on :8100 — start it in another terminal: ./start_sim_docker.sh"
    echo "         (the backend will start anyway and connect once the sim is up)"
fi

echo "Starting backend + GUI on :7860..."
stop_backend
if [ "$MODE" = "--no-agent" ] || [ "$MODE" = "no-agent" ]; then export PHYSYK_AGENTIC=0; fi
nohup "$PY" "$WS/physyk_main_server.py" --port 7860 > "$LOG_DIR/physyk_server.log" 2>&1 &
for i in $(seq 1 30); do healthy http://localhost:7860/health && break; sleep 1; done

echo
check_status
echo
echo "  GUI:  http://localhost:7860"
echo "  Log:  tail -f $LOG_DIR/physyk_server.log"
