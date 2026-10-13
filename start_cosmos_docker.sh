#!/usr/bin/env bash
# Cosmos 3 Nano Reasoner (8B) NIM on :8001 — self-hosted next to Isaac Sim on the same GPU.
#   ./start_cosmos_docker.sh
# Weights download on first start into ~/.cache/nim (several GB, one time).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
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
