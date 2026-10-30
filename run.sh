#!/usr/bin/env bash
# ==============================================================================
# Physyk AI — local runner: starts the BACKEND + GUI only.
#   Backend+GUI: .venv python, http://localhost:7860
#   Isaac Sim  : ./run.sh --sim     (Docker, headless, API on :8100; Ctrl+C stops it)
#   Cosmos     : ./run.sh --cosmos  (optional NIM on :8001; image checks only)
#   Nemotron   : any OpenAI-compatible endpoint: NVIDIA hosted API or Nebius Token Factory (keys in .env, see .env.example)
#
#   ./run.sh              (re)start the backend + GUI
#   ./run.sh --no-agent   same, but every instruction goes straight to Isaac Sim (no LLM calls)
#   ./run.sh --status     health of everything
#   ./run.sh --stop       stop the backend (containers: Ctrl+C the sim terminal / docker stop physyk-cosmos)
#   HEADLESS=0 ./run.sh --sim   open an Isaac Sim window on the host display via X11
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

start_sim() {
    IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
    NAME=physyk-isaac-sim

    docker rm -f "$NAME" >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true   # let the container talk to the host X server

    mkdir -p "$ROOT/.isaac-cache"/{kit,ov,pip,glcache,computecache} "$ROOT/logs"
    # Container user is not the host user. Files the (root) container created earlier can't be
    # chmod'ed by us — that's fine, root can already use them — so never abort on this.
    chmod -R a+rwX "$ROOT/.isaac-cache" "$ROOT/logs" 2>/dev/null || true

    ENV_ARGS=()
    [ -f "$ROOT/.env" ] && ENV_ARGS=(--env-file "$ROOT/.env")   # NVIDIA_API_KEY etc. for perception (Cosmos)

    # Ctrl+C / kill: stop the container (the sim inside does not react to SIGINT on its own), then leave.
    cleanup() { trap - INT TERM; echo; echo "Stopping $NAME ..."; docker stop -t 5 "$NAME" >/dev/null 2>&1 || docker rm -f "$NAME" >/dev/null 2>&1 || true; exit 130; }
    trap cleanup INT TERM

    # Run in the background and wait, so the trap above fires immediately on Ctrl+C.
    docker run --name "$NAME" --init --rm --gpus all --network host "${ENV_ARGS[@]}" \
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -e DISPLAY="${DISPLAY:-:0}" -e ISAAC_SIM_HEADLESS="${HEADLESS:-1}" \
      -e ISAAC_SIM_BIND="${ISAAC_SIM_BIND:-127.0.0.1}" \
      -e PERCEPTION_MODE="${PERCEPTION_MODE:-shadow}" \
      ${PHYSYK_ENCODE_EVERY:+-e PHYSYK_ENCODE_EVERY="$PHYSYK_ENCODE_EVERY"} ${PHYSYK_BACKGROUND:+-e PHYSYK_BACKGROUND="$PHYSYK_BACKGROUND"} ${PERCEPTION_DETECTOR:+-e PERCEPTION_DETECTOR="$PERCEPTION_DETECTOR"} ${PERCEPTION_ZCAL:+-e PERCEPTION_ZCAL="$PERCEPTION_ZCAL"} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v "$ROOT/.isaac-cache/kit:/isaac-sim/kit/cache:rw" \
      -v "$ROOT/.isaac-cache/ov:/root/.cache/ov:rw" \
      -v "$ROOT/.isaac-cache/pip:/root/.cache/pip:rw" \
      -v "$ROOT/.isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
      -v "$ROOT/.isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
      -v "$ROOT/workspace:/workspace:rw" \
      -w /workspace --entrypoint /isaac-sim/python.sh \
      "$IMAGE" isaac_sim_service.py &
    wait $! || true
}

start_cosmos() {
    # Cosmos 3 Nano Reasoner (8B) NIM on :8001. Weights download on first start into ~/.cache/nim (about 10 GB).
    export NGC_API_KEY="${NGC_API_KEY:-${COSMOS_API_KEY:-}}"
    CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
    mkdir -p "$CACHE"; chmod -R a+w "$CACHE" 2>/dev/null || true
    docker rm -f physyk-cosmos >/dev/null 2>&1 || true
    # Lightweight config so it shares the 24 GB GPU with Isaac Sim (weights alone are ~17 GB):
    #  - context 8192 tokens (default 262144 needs 36 GB of KV cache and crashes at startup);
    #    our requests are one image + a short prompt.
    #  - claim at most ~65% of the GPU (FP8 weights 10.3 GB + ~3 GB profiling peak + KV), leaving ~8 GB for the sim.
    # FP8 weights (~9 GB vs ~17 GB bf16). The container only auto-selects bf16 on this GPU, so the
    # generic L40S FP8 nano profile is forced. Set COSMOS_PROFILE= (empty) to use the auto/bf16 one.
    COSMOS_PROFILE="${COSMOS_PROFILE-17fedc428e5a9220fae87b540fc9f324eb9e521d35d733de5fe87253db14e6e7}"
    PROFILE_ENV=""; [ -n "$COSMOS_PROFILE" ] && PROFILE_ENV="-e NIM_MODEL_PROFILE=$COSMOS_PROFILE"
    LIGHT_ENV="$PROFILE_ENV -e NIM_MAX_MODEL_LEN=${COSMOS_MAX_MODEL_LEN:-8192} -e NIM_GPU_MEMORY_UTILIZATION=${COSMOS_GPU_UTIL:-0.65}"
    # NIM_DISABLE_CUDA_GRAPH: CUDA-graph capture crashes on this laptop GPU (Triton kernel launch during capture);
    # eager mode also uses less memory. Fine for occasional single-image calls.
    exec docker run --name physyk-cosmos --rm --gpus all --ipc host --shm-size=8GB \
      -e NGC_API_KEY $LIGHT_ENV -e NIM_DISABLE_CUDA_GRAPH=1 ${COSMOS_NIM_ENV:-} \
      -v "$CACHE:/opt/nim/.cache" -u "$(id -u)" \
      -p 8001:8000 nvcr.io/nim/nvidia/cosmos3-reasoner:latest
}

case "$MODE" in
    --sim|sim)       start_sim; exit 0 ;;
    --cosmos|cosmos) start_cosmos; exit 0 ;;
    --stop|stop)     stop_all; exit 0 ;;
    --status|status) check_status; exit 0 ;;
esac

[ -x "$PY" ] || { echo "Missing $PY — create it: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

if healthy http://localhost:8100/health; then
    echo "Isaac Sim is up on :8100."
else
    echo "WARNING: Isaac Sim is not running on :8100 — start it in another terminal: ./run.sh --sim"
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
