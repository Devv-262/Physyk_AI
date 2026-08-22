#!/bin/bash
# Launch Standalone Isaac Sim 6.0.1 natively with WebRTC streaming and Public IP

ISAAC_ROOT="/opt/dlami/nvme/isaac-sim"

PUBLIC_IP=$(curl -s ifconfig.me || echo "127.0.0.1")

echo "═════════════════════════════════════════════════════════"
echo "  Starting Native Isaac Sim 6.0.1 with WebRTC Streaming  "
echo "  Public IP: $PUBLIC_IP                                  "
echo "  Signaling Port: 49100 | Stream Port: 47998             "
echo "═════════════════════════════════════════════════════════"

export PRIVACY_CONSENT=Y
export ACCEPT_EULA=Y

exec "$ISAAC_ROOT/isaac-sim.streaming.sh" --no-window \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp="$PUBLIC_IP" \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998 \
  "$@"
