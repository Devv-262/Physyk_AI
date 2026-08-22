#!/bin/bash
# Milestone 1c: Read the actual policy examples code and cumotion robot configs
CONTAINER="tender_cerf"

echo "=== isaacsim.robot.policy.examples Python sources ==="
docker exec $CONTAINER find /isaac-sim/exts/isaacsim.robot.policy.examples -name "*.py" 2>/dev/null | head -20

echo ""
echo "=== isaacsim.robot.policy.examples/data ==="
docker exec $CONTAINER ls /isaac-sim/exts/isaacsim.robot.policy.examples/data/ 2>/dev/null

echo ""
echo "=== cumotion robot_configurations ==="
docker exec $CONTAINER ls /isaac-sim/exts/isaacsim.robot_motion.cumotion/robot_configurations/ 2>/dev/null | head -20

echo ""
echo "=== manipulators.examples Python sources ==="
docker exec $CONTAINER find /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples -name "*.py" | head -10

echo ""
echo "=== Sample policy example script content ==="
POLICY_PY=$(docker exec $CONTAINER find /isaac-sim/exts/isaacsim.robot.policy.examples/isaacsim -name "*.py" | head -1)
echo "File: $POLICY_PY"
docker exec $CONTAINER cat "$POLICY_PY" 2>/dev/null | head -60

echo ""
echo "=== Checking if GR00T N1 pip package can be installed inside container ==="
docker exec $CONTAINER /isaac-sim/python.sh -m pip index versions isaac-gr00t 2>&1 | head -5 || echo "pip index not available"
docker exec $CONTAINER /isaac-sim/python.sh -m pip show isaac-gr00t 2>&1 | head -5

echo "=== DONE ==="
