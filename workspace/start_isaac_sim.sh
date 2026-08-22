#!/bin/bash
# Start Isaac Sim with WebRTC streaming

CACHE_ROOT="/opt/dlami/nvme/physyk/cache"
WORKSPACE_ROOT="/opt/dlami/nvme/physyk/workspace"
MODELS_ROOT="/opt/dlami/nvme/physyk/models"

DOCKER_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"

echo "════════════════════════════════════════════"
echo "  Starting Isaac Sim with WebRTC Streaming"
echo "════════════════════════════════════════════"
echo ""

docker run -d \
  --name physyk-isaac-sim \
  --rm \
  --gpus all \
  --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v "$CACHE_ROOT/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_ROOT/ov:/root/.cache/ov:rw" \
  -v "$CACHE_ROOT/pip:/root/.cache/pip:rw" \
  -v "$CACHE_ROOT/glcache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_ROOT/documents:/root/Documents:rw" \
  -v "$WORKSPACE_ROOT:/workspace:rw" \
  -v "$MODELS_ROOT:/models:rw" \
  "$DOCKER_IMAGE" \
  ./runheadless.webrtc.sh

