#!/bin/bash
CONTAINER=$(docker ps --filter "ancestor=nvcr.io/nvidia/isaac-sim:6.0.1" --filter "status=running" --format "{{.Names}}" | head -1)
echo "=== Container: $CONTAINER ==="

echo "=== Stopping old python.sh processes ==="
docker exec "$CONTAINER" bash -c "pkill -9 -f physyk_milestone1 2>/dev/null || true"
sleep 2

echo "=== Copying updated script into container ==="
docker cp /opt/dlami/nvme/physyk/workspace/physyk_milestone1.py "$CONTAINER":/tmp/physyk_milestone1.py
docker exec "$CONTAINER" ls -la /tmp/physyk_milestone1.py

echo "=== Clearing old log ==="
docker exec "$CONTAINER" bash -c "rm -f /tmp/milestone1.log"

echo "=== Launching via /isaac-sim/python.sh ==="
docker exec "$CONTAINER" bash -c \
    "nohup /isaac-sim/python.sh /tmp/physyk_milestone1.py > /tmp/milestone1.log 2>&1 &"

echo "=== Waiting 55s for full kit startup + physics ==="
sleep 55

echo "=== Output (last 40 lines) ==="
docker exec "$CONTAINER" cat /tmp/milestone1.log | grep -E "Physyk|Error|Traceback|step=|COMPLETE|startup|Startup" | tail -40

echo ""
echo "=== Is python.sh still running? ==="
docker exec "$CONTAINER" ps aux | grep python | grep -v grep || echo "Completed"

docker cp "$CONTAINER":/tmp/milestone1.log /opt/dlami/nvme/physyk/milestone1.log 2>/dev/null
echo "=== DONE ==="
