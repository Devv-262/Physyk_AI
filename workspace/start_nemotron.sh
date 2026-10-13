#!/bin/bash
# ==============================================================================
# Physyk AI — Nemotron-30B-FP8 Local vLLM Inference Server
# System 2 Cognitive Brain for Franka Panda Robot & Multimodal VLM
# Bound to 0.0.0.0:8000 for local access & Brev Dashboard port forwarding
# ==============================================================================

set -e

VENV_PYTHON="/opt/dlami/nvme/physyk/workspace/.venv-nemotron/bin/python3"
MODEL_DIR="/opt/dlami/nvme/physyk/models/Nemotron-30B-FP8"
LOG_FILE="/opt/dlami/nvme/physyk/logs/nemotron_server.log"
PID_FILE="/opt/dlami/nvme/physyk/logs/nemotron_server.pid"

mkdir -p /opt/dlami/nvme/physyk/logs

echo "======================================================================"
echo "  [Physyk Nemotron] Starting vLLM Server on 0.0.0.0:8000"
echo "  [Physyk Nemotron] Model: $MODEL_DIR"
echo "  [Physyk Nemotron] Hardware: RTX PRO 6000 GPU (Max VRAM allocation: ~35 GB)"
echo "======================================================================"

# Kill any previous Nemotron instance only — scoped to this model's own path so a
# separately-running Cosmos-Reason2 vLLM server (port 8001) is never collaterally killed
# (previously matched ANY "vllm.entrypoints.openai" process, which is exactly what caused a
# real Cosmos outage earlier this session when Nemotron was restarted).
pkill -f "vllm.entrypoints.openai.*Nemotron-30B" 2>/dev/null || true
pkill -f "vllm serve.*Nemotron-30B" 2>/dev/null || true
sleep 1

# Clear the "intentionally stopped" sentinel (set by run.sh --stop) — this script is
# about to bring Nemotron back up, so nemotron_watchdog.sh should resume watching it.
rm -f /opt/dlami/nvme/physyk/logs/.nemotron_intentionally_stopped

# Launch vLLM server
# --max-model-len MUST stay >= the orchestrator's largest single-call max_tokens (4096,
# see physyk_agent_orchestrator.py's _call_nemotron) plus real prompt-token headroom — a
# request needs (prompt_tokens + max_tokens) <= max_model_len or vLLM 400s outright,
# with ZERO room for the prompt at max_tokens==max_model_len. This was previously left at
# 4096 here (matching neither that code's own comment nor its actual max_tokens value),
# which silently 400'd every single decompose/diagnose call — confirmed live 2026-08-19.
exec $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name nemotron \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.55 \
  --trust-remote-code \
  --no-enable-log-requests
