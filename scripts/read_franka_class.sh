#!/bin/bash
CONTAINER="tender_cerf"
echo "=== franka.py full source ==="
docker exec "$CONTAINER" cat /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/isaacsim/robot/experimental/manipulators/examples/franka/franka.py
echo "=== DONE ==="
