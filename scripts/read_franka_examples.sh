#!/bin/bash
CONTAINER="tender_cerf"

echo "=== pick_place.py full content ==="
docker exec $CONTAINER cat /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/isaacsim/robot/experimental/manipulators/examples/franka/pick_place.py

echo ""
echo "=== franka.py (manipulators) full content ==="
docker exec $CONTAINER cat /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/isaacsim/robot/experimental/manipulators/examples/franka/franka.py

echo ""
echo "=== franka_example.py (policy runner) full content ==="
docker exec $CONTAINER cat /isaac-sim/exts/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/franka/franka_example.py

echo "=== DONE ==="
