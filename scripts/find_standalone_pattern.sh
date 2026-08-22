#!/bin/bash
CONTAINER="tender_cerf"
echo "=== interactive pick_place extension source ==="
docker exec "$CONTAINER" cat /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples/isaacsim/robot/experimental/manipulators/examples/interactive/pick_place/__init__.py 2>/dev/null

echo ""
echo "=== robo_factory extension (the headless runner) ==="
docker exec "$CONTAINER" find /isaac-sim/exts/isaacsim.robot.experimental.manipulators.examples -name "*.py" | xargs grep -l "headless\|SimulationApp\|run_loop\|while" 2>/dev/null | head -5

echo ""
echo "=== standalone_examples for manipulators ==="
docker exec "$CONTAINER" find /isaac-sim -path "*/standalone_examples/*" -name "*.py" 2>/dev/null | grep -i "franka\|pick\|manipulat" | head -10

echo ""
echo "=== standalone_examples root ==="
docker exec "$CONTAINER" ls /isaac-sim/standalone_examples/ 2>/dev/null | head -20
echo "=== DONE ==="
