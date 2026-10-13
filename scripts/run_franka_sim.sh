#!/bin/bash
CONTAINER=$(docker ps --filter "ancestor=nvcr.io/nvidia/isaac-sim:6.0.1" --filter "status=running" --format "{{.Names}}" | head -1)
echo "Using Isaac Sim Container: $CONTAINER"

# Copy python script into container
docker cp /opt/dlami/nvme/physyk/workspace/physyk_franka_sim.py "$CONTAINER":/tmp/physyk_franka_sim.py

# Execute directly with python.sh
echo "Running /isaac-sim/python.sh /tmp/physyk_franka_sim.py ..."
docker exec "$CONTAINER" /isaac-sim/python.sh /tmp/physyk_franka_sim.py
