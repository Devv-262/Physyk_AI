#!/bin/bash
CONTAINER="tender_cerf"
echo "=== AppFramework signature ==="
docker exec "$CONTAINER" /isaac-sim/python.sh -c "
from isaacsim import AppFramework
import inspect
print(inspect.signature(AppFramework.__init__))
print('---')
help(AppFramework.__init__)
" 2>&1 | head -40

echo ""
echo "=== Check SimulationApp (alternative entry point) ==="
docker exec "$CONTAINER" /isaac-sim/python.sh -c "
from isaacsim.core.api import SimulationApp
import inspect
print(inspect.signature(SimulationApp.__init__))
" 2>&1 | head -20

echo ""
echo "=== Check isaacsim.SimulationApp ==="
docker exec "$CONTAINER" /isaac-sim/python.sh -c "
import isaacsim.SimulationApp as sa
print(dir(sa))
" 2>&1 | head -10

echo "=== DONE ==="
