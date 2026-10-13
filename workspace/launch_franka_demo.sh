#!/bin/bash
# ============================================================
#  launch_franka_demo.sh
#  Starts Isaac Sim 6.0.1 container and runs the Franka demo
#  with WebRTC streaming on port 8211
# ============================================================

set -e

CACHE_ROOT="/opt/dlami/nvme/physyk/cache"
WORKSPACE_ROOT="/opt/dlami/nvme/physyk/workspace"
MODELS_ROOT="/opt/dlami/nvme/physyk/models"
DOCKER_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"
CONTAINER_NAME="physyk-franka-demo"

# Clean up any stale container
docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "[INFO] Removed stale container: $CONTAINER_NAME" || true

mkdir -p "$CACHE_ROOT/kit" "$CACHE_ROOT/ov" "$CACHE_ROOT/pip" \
         "$CACHE_ROOT/glcache" "$CACHE_ROOT/logs" "$CACHE_ROOT/data" \
         "$CACHE_ROOT/documents"

echo ""
echo "============================================================"
echo "  Franka Panda — Isaac Sim 6.0.1 WebRTC Demo"
echo "============================================================"
echo ""
echo "  📡 After ~60s startup, open your browser at:"
echo "     http://localhost:8211/streaming/webrtc-demo/"
echo ""
echo "  🎥 You will see:"
echo "     - Scene camera  : angled third-person view"
echo "     - Wrist camera  : eye-in-hand mounted on link8"
echo "     - Full pick-and-place motion sequence"
echo ""
echo "============================================================"
echo ""

docker run \
    --name "$CONTAINER_NAME" \
    --rm \
    --gpus all \
    --network host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -v "$CACHE_ROOT/kit:/isaac-sim/kit/cache:rw" \
    -v "$CACHE_ROOT/ov:/root/.cache/ov:rw" \
    -v "$CACHE_ROOT/pip:/root/.cache/pip:rw" \
    -v "$CACHE_ROOT/glcache:/root/.nv/ComputeCache:rw" \
    -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "$CACHE_ROOT/data:/root/.local/share/ov/data:rw" \
    -v "$CACHE_ROOT/documents:/root/Documents:rw" \
    -v "$WORKSPACE_ROOT:/workspace:rw" \
    -v "$MODELS_ROOT:/models:rw" \
    -p 8211:8211 \
    -p 47995-48012:47995-48012/udp \
    -p 47995-48012:47995-48012/tcp \
    -p 49000-49007:49000-49007/udp \
    -p 49000-49007:49000-49007/tcp \
    "$DOCKER_IMAGE" \
    bash -c "
        echo '[Container] Starting WebRTC streaming service...'
        # Start the WebRTC server in background
        /isaac-sim/runheadless.webrtc.sh &
        WEBRTC_PID=\$!
        sleep 45
        echo '[Container] Launching Franka Panda demo script...'
        /isaac-sim/python.sh /workspace/franka_isaac_demo.py
        kill \$WEBRTC_PID 2>/dev/null || true
    "
