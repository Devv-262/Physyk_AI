#!/bin/bash
# Milestone 1: Check Isaac Sim container internals for GR00T and LIBERO Panda USD assets
CONTAINER=$(docker ps --filter "name=tender_cerf" --format "{{.Names}}" | head -1)
echo "=== Using container: $CONTAINER ==="

echo ""
echo "=== GR00T extension search ==="
docker exec $CONTAINER find /isaac-sim/extscache -maxdepth 2 -name "*.groot*" -o -name "*groot*" 2>/dev/null | head -20 || echo "nothing in extscache"

echo ""
echo "=== isaacsim.robot extensions ==="
docker exec $CONTAINER ls /isaac-sim/exts/ | grep -i robot

echo ""
echo "=== Franka / Panda USD robot assets ==="
docker exec $CONTAINER find /isaac-sim -name "*.usd" 2>/dev/null | grep -i -E "franka|panda|libero" | head -20

echo ""
echo "=== isaacsim.asset paths ==="
docker exec $CONTAINER bash -c "cat /isaac-sim/extscache/isaacsim.asset.browser.robots*/config/extension.toml 2>/dev/null | head -20 || echo not-found"

echo ""
echo "=== isaac-sim python.sh available packages ==="
docker exec $CONTAINER /isaac-sim/python.sh -c "import isaacsim; print('isaacsim OK')" 2>&1 | tail -5

echo "=== DONE ==="
