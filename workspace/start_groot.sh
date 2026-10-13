#!/bin/bash
# ==============================================================================
# Physyk AI — GR00T-N1.7 LIBERO Policy Server (Phase 3 real VLA inference)
# System 1 Sensorimotor Brain — real Gr00tPolicy forward passes, not a stub.
# Runs under Isaac-GR00T/.venv (the only environment with the pinned
# torch/transformers/flash-attn build this checkpoint needs — Isaac Sim's own
# bundled Python does not have these). isaac_sim_service.py is an HTTP client
# of this server for any "vla: <instruction>" command.
# ==============================================================================

set -e

VENV_PYTHON="/opt/dlami/nvme/physyk/workspace/Isaac-GR00T/.venv/bin/python3"
LOG_FILE="/opt/dlami/nvme/physyk/logs/groot_server.log"

# Fine-tuned on 54 red-cube-only episodes collected from THIS scene
# (finetune_data/raw_episodes_red, converted via scripts/convert_raw_episodes_to_lerobot.py)
# — see finetune_data/checkpoints/physyk_red_54 for the training run (train_loss 0.0402,
# 1500 steps). Supersedes the earlier 21-episode red-only checkpoint
# (finetune_data/checkpoints/physyk_red_21) and the original 17-episode mixed-color
# checkpoint (finetune_data/checkpoints/physyk_cubes_17), both kept on disk but no longer
# the default. Falls back to the base LIBERO checkpoint if the fine-tuned one isn't present
# (e.g. a fresh checkout that hasn't run the fine-tune yet). Override with
# GROOT_MODEL_PATH=... to pick a specific checkpoint.
DEFAULT_FINETUNED="/opt/dlami/nvme/physyk/workspace/finetune_data/checkpoints/physyk_red_54/checkpoint-1500"
if [ -z "${GROOT_MODEL_PATH:-}" ]; then
    if [ -d "$DEFAULT_FINETUNED" ]; then
        export GROOT_MODEL_PATH="$DEFAULT_FINETUNED"
    else
        export GROOT_MODEL_PATH="/opt/dlami/nvme/physyk/models/GR00T-N1.7-LIBERO/libero_10"
    fi
fi

mkdir -p /opt/dlami/nvme/physyk/logs

echo "======================================================================"
echo "  [Physyk GR00T] Starting GR00T-N1.7 Policy Server on :8300"
echo "  [Physyk GR00T] Checkpoint: $GROOT_MODEL_PATH"
echo "======================================================================"

pkill -f "groot_policy_service.py" 2>/dev/null || true
sleep 1

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"

exec "$VENV_PYTHON" /opt/dlami/nvme/physyk/workspace/groot_policy_service.py
