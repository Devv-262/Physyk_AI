#!/bin/bash
# Find the correct kit experience file that includes robot extensions
CONTAINER="tender_cerf"

echo "=== Available kit experience files ==="
docker exec "$CONTAINER" ls /isaac-sim/apps/*.kit

echo ""
echo "=== isaacsim.exp.full.kit dependencies (does it include robot.experimental?) ==="
docker exec "$CONTAINER" grep -r "robot.experimental\|manipulator" /isaac-sim/apps/ 2>/dev/null | head -20

echo ""
echo "=== Check what SimulationApp actually loads by default ==="
docker exec "$CONTAINER" cat /isaac-sim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py 2>/dev/null | grep -A 5 "exp.base\|experience\|kit_file" | head -30

echo ""
echo "=== isaacsim.robot.experimental.manipulators extension folder ==="
docker exec "$CONTAINER" ls /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/config/

echo ""
echo "=== extension.toml for manipulators ==="
docker exec "$CONTAINER" cat /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/config/extension.toml

echo "=== DONE ==="
