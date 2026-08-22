#!/bin/bash
# Physyk AI — Isaac Sim Launcher (RTX PRO 6000 Blackwell)

MODE="${1:-headless}"
ACCEPT_EULA="Y"

CACHE_ROOT="/opt/dlami/nvme/physyk/cache"
WORKSPACE_ROOT="/opt/dlami/nvme/physyk/workspace"
MODELS_ROOT="/opt/dlami/nvme/physyk/models"
SCRIPTS_ROOT="/opt/dlami/nvme/physyk/scripts"

mkdir -p "$CACHE_ROOT/kit" "$CACHE_ROOT/ov" "$CACHE_ROOT/pip" "$CACHE_ROOT/glcache" \
         "$CACHE_ROOT/logs" "$CACHE_ROOT/data" \
         "$CACHE_ROOT/documents" "$WORKSPACE_ROOT" "$MODELS_ROOT" "$SCRIPTS_ROOT"

DOCKER_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"

DOCKER_ARGS=(
    --name physyk-isaac-sim
    --rm
    --gpus all
    --network host
    -e "ACCEPT_EULA=${ACCEPT_EULA}"
    -e "PRIVACY_CONSENT=Y"
    -v "$CACHE_ROOT/kit:/isaac-sim/kit/cache:rw"
    -v "$CACHE_ROOT/ov:/root/.cache/ov:rw"
    -v "$CACHE_ROOT/pip:/root/.cache/pip:rw"
    -v "$CACHE_ROOT/glcache:/root/.nv/ComputeCache:rw"
    -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs:rw"
    -v "$CACHE_ROOT/data:/root/.local/share/ov/data:rw"
    -v "$CACHE_ROOT/documents:/root/Documents:rw"
    -v "$WORKSPACE_ROOT:/workspace:rw"
    -v "$MODELS_ROOT:/models:rw"
)

if [ "$MODE" == "webrtc" ]; then
    echo "=== Launching Physyk AI Simulation with WebRTC Streaming on port 8211 ==="
    echo "Open browser: http://<VM_IP>:8211/streaming/webrtc-demo/?server=<VM_IP>"
    docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" ./runheadless.webrtc.sh
elif [ "$MODE" == "native" ]; then
    echo "=== Launching Physyk AI Simulation with Omniverse Streaming Client ==="
    echo "Connect Omniverse Streaming Client to <VM_IP>"
    docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" ./runheadless.native.sh
elif [ "$MODE" == "bash" ]; then
    echo "=== Launching interactive bash inside Isaac Sim container ==="
    docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" bash
else
    echo "=== Launching Physyk AI Simulation Headless ==="
    docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" ./runheadless.sh "$@"
fi
