#!/bin/bash
set -e

CACHE_ROOT="/opt/dlami/nvme/physyk/cache"
WORKSPACE_ROOT="/opt/dlami/nvme/physyk/workspace"
MODELS_ROOT="/opt/dlami/nvme/physyk/models"

mkdir -p "$CACHE_ROOT/kit" "$CACHE_ROOT/ov" "$CACHE_ROOT/pip" "$CACHE_ROOT/glcache" \
         "$CACHE_ROOT/logs" "$CACHE_ROOT/data" \
         "$CACHE_ROOT/documents" "$WORKSPACE_ROOT" "$MODELS_ROOT"

echo "=== Running Physyk AI Franka Assembly Digital Twin Test ==="
docker run --rm --gpus all --network host \
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
  nvcr.io/nvidia/isaac-sim:6.0.1 ./python.sh /workspace/test_physyk_franka.py
