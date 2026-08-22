#!/bin/bash
# Milestone 1b: Find Franka USD assets and GR00T in Isaac Sim extscache
CONTAINER="tender_cerf"
echo "=== Searching extscache for Franka/Panda/Manipulator ==="
docker exec $CONTAINER find /isaac-sim/extscache -maxdepth 3 -name "*.toml" 2>/dev/null | xargs grep -l -i "franka\|panda\|manipulator" 2>/dev/null | head -20

echo ""
echo "=== Searching extscache for GR00T ==="
docker exec $CONTAINER find /isaac-sim/extscache -maxdepth 3 -name "*.toml" 2>/dev/null | xargs grep -l -i "groot\|gr00t" 2>/dev/null | head -10

echo ""
echo "=== isaacsim.robot.experimental.manipulators.examples content ==="
docker exec $CONTAINER ls /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/ 2>/dev/null

echo ""
echo "=== isaacsim.robot.policy.examples content ==="
docker exec $CONTAINER ls /isaac-sim/exts/isaacsim.robot.policy.examples/ 2>/dev/null

echo ""
echo "=== cuMotion extension (real IK/motion planning) ==="
docker exec $CONTAINER ls /isaac-sim/exts/isaacsim.robot_motion.cumotion/ 2>/dev/null

echo ""
echo "=== Isaac Sim Python: test robot USD path from nucleus ==="
docker exec $CONTAINER /isaac-sim/python.sh -c "
import carb
import omni.kit.app
print(carb.settings.get_settings().get('/persistent/isaac/asset_root/default') or 'no-nucleus-default')
" 2>&1 | tail -3

echo ""
echo "=== Check if LIBERO asset pack is locally installed ==="
docker exec $CONTAINER find /isaac-sim -name "*.usd" 2>/dev/null | head -30

echo "=== DONE ==="
