#!/bin/bash
CONTAINER="tender_cerf"
echo "=== control_frankas.py FULL ==="
docker exec "$CONTAINER" cat /isaac-sim/standalone_examples/api/isaacsim.core.experimental.api/control_frankas.py

echo ""
echo "=== tutorial_9_gripper_control.py (simplest Franka standalone) ==="
docker exec "$CONTAINER" cat /isaac-sim/standalone_examples/tutorials/manipulation/tutorial_9_gripper_control.py

echo "=== DONE ==="
