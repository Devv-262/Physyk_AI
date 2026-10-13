#!/bin/bash
CONTAINER=$(docker ps --filter "ancestor=nvcr.io/nvidia/isaac-sim:6.0.1" --filter "status=running" --format "{{.Names}}" | head -1)
echo "Container: $CONTAINER"
echo ""
echo "=== Full log filtered to Physyk lines and errors ==="
docker exec "$CONTAINER" grep -E "Physyk|Error \[omni.kit.app|Traceback|step=|COMPLETE|app ready|Startup Complete" /tmp/milestone1.log 2>/dev/null

echo ""
echo "=== Last 20 lines of raw log ==="
docker exec "$CONTAINER" tail -20 /tmp/milestone1.log 2>/dev/null

echo ""
echo "=== Python process status ==="
docker exec "$CONTAINER" ps aux | grep python | grep -v grep || echo "Completed"
