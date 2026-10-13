#!/usr/bin/env bash
set -euo pipefail
WS=/opt/dlami/nvme/physyk/workspace
PORT=7860

echo "========================================"
echo "  PHYSYK — Franka Panda Physical AI"
echo "  Port: $PORT"
echo "========================================"

# Kill stale
pkill -f physyk_main_server || true
pkill -f cloudflared        || true
pkill -f pinggy             || true
sleep 1

# Start server
cd "$WS"
nohup python3 physyk_main_server.py --port $PORT > /tmp/physyk_server.log 2>&1 &
echo "Server PID: $!"

# Wait for server
echo -n "Waiting for server..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    echo " OK"; break
  fi
  sleep 1; echo -n "."
done

# Start tunnel
echo "Starting Cloudflare tunnel..."
nohup cloudflared tunnel --url http://localhost:$PORT --no-autoupdate \
  > /tmp/physyk_tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "Tunnel PID: $TUNNEL_PID"

# Extract URL
echo -n "Waiting for public URL"
URL=""
for i in $(seq 1 40); do
  URL=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' /tmp/physyk_tunnel.log 2>/dev/null | head -1)
  if [ -n "$URL" ]; then break; fi
  sleep 1; echo -n "."
done

echo ""
if [ -n "$URL" ]; then
  echo "========================================"
  echo "  PUBLIC URL: $URL"
  echo "========================================"
  # Try to open browser
  xdg-open "$URL" 2>/dev/null || true
else
  echo "Tunnel URL not obtained. Check /tmp/physyk_tunnel.log"
  echo "Server is at http://localhost:$PORT"
fi
