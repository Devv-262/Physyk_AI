#!/bin/bash
CONTAINER="tender_cerf"

echo "=== tutorial_9_pick_place_cumotion.py ==="
docker exec "$CONTAINER" cat /isaac-sim/standalone_examples/tutorials/manipulation/tutorial_9_pick_place_cumotion.py 2>/dev/null | head -120

echo ""
echo "=== control_frankas.py (core API example) ==="
docker exec "$CONTAINER" cat /isaac-sim/standalone_examples/api/isaacsim.core.experimental.api/control_frankas.py 2>/dev/null | head -120

echo "=== DONE ==="
