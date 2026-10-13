#!/bin/bash
# ===================================================================
# Physyk AI — Single One-Click Pipeline Launcher (Isaac Sim 6.0.1)
# ===================================================================

set -e

WORKSPACE_DIR="/opt/dlami/nvme/physyk"
LOG_DIR="$WORKSPACE_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$WORKSPACE_DIR/workspace" "$LOG_DIR"

echo "=================================================================="
echo "  [Physyk AI] Launching Isaac Sim Franka Pipeline on Blackwell GPU"
echo "=================================================================="
echo "  Timestamp : $(date)"
echo "  Log File  : $LOG_FILE"
echo "------------------------------------------------------------------"

# 1. Verify GPU
echo "[1/4] Checking GPU status..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Verify or Pick Isaac Sim 6.0.1 Docker Container
echo "[2/4] Checking Isaac Sim Docker container..."
CONTAINER=$(docker ps --filter "ancestor=nvcr.io/nvidia/isaac-sim:6.0.1" --filter "status=running" --format "{{.Names}}" | head -1)

if [ -z "$CONTAINER" ]; then
    echo "      No running Isaac Sim container found. Starting fresh container..."
    CONTAINER="isaac_sim_physyk"
    docker run -d --name "$CONTAINER" \
        --entrypoint bash \
        --gpus all \
        -e "ACCEPT_EULA=Y" \
        --network=host \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /opt/dlami/nvme/physyk:/opt/physyk \
        nvcr.io/nvidia/isaac-sim:6.0.1 \
        -c "sleep infinity"
    echo "      Started container: $CONTAINER"
else
    echo "      Using active container: $CONTAINER"
fi

# 3. Deploy Python Pipeline Script into Container
echo "[3/4] Deploying simulation script..."
docker cp "$WORKSPACE_DIR/workspace/physyk_pipeline.py" "$CONTAINER":/tmp/physyk_pipeline.py

# 4. Execute Simulation with Live Output Streaming (tee)
echo "[4/4] Executing Isaac Sim Pipeline (Streaming Live Output)..."
echo "=================================================================="

# Run python.sh inside container and pipe through tee to show on terminal AND save to log file
docker exec -i "$CONTAINER" /isaac-sim/python.sh /tmp/physyk_pipeline.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================================="
echo "  [DONE] Pipeline execution completed successfully!"
echo "  Full log saved to: $LOG_FILE"
echo "=================================================================="
