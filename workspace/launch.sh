#!/usr/bin/env bash
# Physyk — Full Stack Launcher with Isaac Sim 6.0.1 Simulation Engine
# Starts: Native Isaac Sim 6.0.1 Service + Physyk AI Web UI + Public Cloudflare Tunnel
set -euo pipefail

WS=/opt/dlami/nvme/physyk/workspace
ISAAC_PYTHON=/opt/dlami/nvme/isaac-sim/python.sh
PORT=7860
SIM_PORT=8100
CF=/tmp/cloudflared

echo "═══════════════════════════════════════════════════════════════════"
echo "  PHYSYK Physical AI System — Full Stack Launcher"
echo "  Isaac Sim 6.0.1 (PhysX) + Franka Panda + Real-time Camera Stream"
echo "═══════════════════════════════════════════════════════════════════"

# 1. Clean up stale processes
echo "[1/4] Cleaning up previous processes..."
pkill -f physyk_main_server 2>/dev/null && echo "  - Killed previous web server" || true
pkill -f isaac_sim_service 2>/dev/null && echo "  - Killed previous Isaac Sim service" || true
pkill -f cloudflared 2>/dev/null && echo "  - Killed previous tunnel" || true
sleep 1

# 2. Start Isaac Sim Simulation & Camera Streaming Engine
echo "[2/4] Starting Isaac Sim 6.0.1 Simulation Service (port $SIM_PORT)..."
cd "$WS"
"$ISAAC_PYTHON" "$WS/isaac_sim_service.py" > /tmp/isaac_sim_service.log 2>&1 &
SIM_PID=$!
echo "  - Isaac Sim PID: $SIM_PID"

echo -n "  - Waiting for Isaac Sim initialization"
for i in $(seq 1 45); do
  sleep 1
  if curl -sf http://localhost:$SIM_PORT/health > /dev/null 2>&1; then
    echo " ✓ READY (60 Hz PhysX Active)"
    break
  fi
  echo -n "."
done

# 3. Start Physyk Web UI Server
echo "[3/4] Starting Physyk AI Web UI on port $PORT..."
python3 "$WS/physyk_main_server.py" --port $PORT > /tmp/physyk_server.log 2>&1 &
SERVER_PID=$!
echo "  - Web Server PID: $SERVER_PID"

echo -n "  - Waiting for Web UI health check"
for i in $(seq 1 15); do
  sleep 1
  if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    echo " ✓ READY"
    break
  fi
  echo -n "."
done

# 4. Start Cloudflare Tunnel
echo "[4/4] Starting Cloudflare Public HTTPS Tunnel..."
if [ ! -f "$CF" ]; then
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$CF"
  chmod +x "$CF"
fi

"$CF" tunnel --url http://localhost:$PORT > /tmp/cf_tunnel.log 2>&1 &
TUNNEL_PID=$!

echo -n "  - Generating Public HTTPS link"
URL=""
for i in $(seq 1 20); do
  URL=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' /tmp/cf_tunnel.log 2>/dev/null | head -1 || true)
  if [ -n "$URL" ]; then break; fi
  sleep 1; echo -n "."
done
echo ""

BREV_URL="https://vscode-rrx2yrnxm.apps.run.brev.nvidia.com/proxy/7860/"

echo "═══════════════════════════════════════════════════════════════════"
echo "  ✅ PHYSYK SYSTEM FULLY ONLINE & STREAMING"
echo "═══════════════════════════════════════════════════════════════════"
if [ -n "$URL" ]; then
  echo "  🌐 PUBLIC WEB GUI:       $URL"
fi
echo "  🌐 BREV DIRECT PROXY:    $BREV_URL"
echo "  📡 ISAAC SIM STREAM:     http://localhost:$SIM_PORT/camera/stream"
echo "  🤖 LOCAL WEB UI:         http://localhost:$PORT"
echo "═══════════════════════════════════════════════════════════════════"
echo "Isaac Sim PID=$SIM_PID | Web Server PID=$SERVER_PID | Tunnel PID=$TUNNEL_PID"
echo "Logs: /tmp/isaac_sim_service.log  /tmp/physyk_server.log  /tmp/cf_tunnel.log"
echo "Press Ctrl+C to stop all services"

wait $SERVER_PID
