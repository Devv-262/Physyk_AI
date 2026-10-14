#!/usr/bin/env bash
# Runs isaac_sim_service.py inside the official Isaac Sim container (Ubuntu 22.04 userland),
# so the host OS (Ubuntu 26.04) doesn't matter.
#   ./start_sim_docker.sh            # headless (no window), API on 127.0.0.1:8100; Ctrl+C stops it
#   HEADLESS=0 ./start_sim_docker.sh # with a window on the host display via X11
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
  ${PHYSYK_PROFILE:+-e PHYSYK_PROFILE="$PHYSYK_PROFILE"} ${PHYSYK_SKIP_ORCH:+-e PHYSYK_SKIP_ORCH="$PHYSYK_SKIP_ORCH"} ${PHYSYK_SPEEDUP:+-e PHYSYK_SPEEDUP="$PHYSYK_SPEEDUP"} ${PHYSYK_SPEEDUP_TRANSIT:+-e PHYSYK_SPEEDUP_TRANSIT="$PHYSYK_SPEEDUP_TRANSIT"} ${PHYSYK_SPEEDUP_VERTICAL:+-e PHYSYK_SPEEDUP_VERTICAL="$PHYSYK_SPEEDUP_VERTICAL"} ${PHYSYK_ENCODE_EVERY:+-e PHYSYK_ENCODE_EVERY="$PHYSYK_ENCODE_EVERY"} ${PHYSYK_ARM_KD:+-e PHYSYK_ARM_KD="$PHYSYK_ARM_KD"} ${PHYSYK_JOINT_TRACE:+-e PHYSYK_JOINT_TRACE="$PHYSYK_JOINT_TRACE"} ${PHYSYK_CUBE_OBSTACLES:+-e PHYSYK_CUBE_OBSTACLES="$PHYSYK_CUBE_OBSTACLES"} ${PHYSYK_ARM_OBSTACLES:+-e PHYSYK_ARM_OBSTACLES="$PHYSYK_ARM_OBSTACLES"} ${PHYSYK_RANDOM_SEED:+-e PHYSYK_RANDOM_SEED="$PHYSYK_RANDOM_SEED"} ${PHYSYK_MIN_RADIUS:+-e PHYSYK_MIN_RADIUS="$PHYSYK_MIN_RADIUS"} ${PHYSYK_BACKGROUND:+-e PHYSYK_BACKGROUND="$PHYSYK_BACKGROUND"} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$ROOT/.isaac-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$ROOT/.isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$ROOT/.isaac-cache/pip:/root/.cache/pip:rw" \
  -v "$ROOT/.isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$ROOT/.isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$ROOT/workspace:/workspace:rw" -v "$ROOT/assets:/assets:ro" \
  -w /workspace --entrypoint /isaac-sim/python.sh \
  "$IMAGE" isaac_sim_service.py "$@" &
wait $! || true
