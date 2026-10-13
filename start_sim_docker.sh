#!/usr/bin/env bash
# Runs isaac_sim_service.py inside the official Isaac Sim container (Ubuntu 22.04 userland),
# so the host OS (Ubuntu 26.04) doesn't matter. Window shows on the host display via X11.
#   ./start_sim_docker.sh            # windowed, API on 127.0.0.1:8100
#   HEADLESS=1 ./start_sim_docker.sh # no window
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

exec docker run --name "$NAME" --rm --gpus all --network host "${ENV_ARGS[@]}" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY="${DISPLAY:-:0}" -e ISAAC_SIM_HEADLESS="${HEADLESS:-0}" \
  -e ISAAC_SIM_BIND="${ISAAC_SIM_BIND:-127.0.0.1}" \
  -e PERCEPTION_MODE="${PERCEPTION_MODE:-shadow}" \
  ${PHYSYK_PROFILE:+-e PHYSYK_PROFILE="$PHYSYK_PROFILE"} ${PHYSYK_SKIP_ORCH:+-e PHYSYK_SKIP_ORCH="$PHYSYK_SKIP_ORCH"} ${PHYSYK_SPEEDUP:+-e PHYSYK_SPEEDUP="$PHYSYK_SPEEDUP"} ${PHYSYK_ARM_KD:+-e PHYSYK_ARM_KD="$PHYSYK_ARM_KD"} ${PHYSYK_BACKGROUND:+-e PHYSYK_BACKGROUND="$PHYSYK_BACKGROUND"} ${PHYSYK_GLASS_OPACITY:+-e PHYSYK_GLASS_OPACITY="$PHYSYK_GLASS_OPACITY"} ${PHYSYK_GLASS_IOR:+-e PHYSYK_GLASS_IOR="$PHYSYK_GLASS_IOR"} ${PHYSYK_GLASS:+-e PHYSYK_GLASS="$PHYSYK_GLASS"} ${PHYSYK_LIQUID_GLOW:+-e PHYSYK_LIQUID_GLOW="$PHYSYK_LIQUID_GLOW"} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$ROOT/.isaac-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$ROOT/.isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$ROOT/.isaac-cache/pip:/root/.cache/pip:rw" \
  -v "$ROOT/.isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$ROOT/.isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$ROOT/workspace:/workspace:rw" -v "$ROOT/assets:/assets:ro" \
  -w /workspace --entrypoint /isaac-sim/python.sh \
  "$IMAGE" isaac_sim_service.py "$@"
