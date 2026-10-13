#!/usr/bin/env bash
# Physyk — Phase 1 Relaunch Script
set -euo pipefail
WS=/opt/dlami/nvme/physyk/workspace
ISAAC_PYTHON=/opt/dlami/nvme/isaac-sim/python.sh

echo "🔴 Stopping stale processes..."
pkill -f physyk_main_server 2>/dev/null || true
pkill -f isaac_sim_service   2>/dev/null || true
sleep 2

echo "🚀 Starting Isaac Sim Service (port 8100)..."
nohup "$ISAAC_PYTHON" "$WS/isaac_sim_service.py" \
  > /tmp/isaac_sim_service.log 2>&1 &
echo "   Isaac Sim PID: $!"

echo "⏳ Waiting for Isaac Sim to initialize (~60s)..."
for i in $(seq 1 90); do
  sleep 1
  if curl -sf http://localhost:8100/health > /dev/null 2>&1; then
    echo "   ✅ Isaac Sim READY (${i}s)"
    break
  fi
  [ $((i % 10)) -eq 0 ] && echo "   ... still starting (${i}s)"
done

echo "🌐 Starting Physyk Web GUI (port 7860)..."
nohup python3 "$WS/physyk_main_server.py" --port 7860 \
  > /tmp/physyk_server.log 2>&1 &
echo "   Web Server PID: $!"

sleep 3
curl -sf http://localhost:7860/health > /dev/null 2>&1 \
  && echo "   ✅ Physyk GUI READY" \
  || echo "   ⚠️  GUI not yet ready — check /tmp/physyk_server.log"

echo ""
echo "════════════════════════════════════"
echo "  🟢 PHYSYK ONLINE"
echo "  GUI:    http://localhost:7860"
echo "  Camera: http://localhost:8100/camera/stream"
echo "  Logs:   /tmp/isaac_sim_service.log"
echo "          /tmp/physyk_server.log"
echo "════════════════════════════════════"
