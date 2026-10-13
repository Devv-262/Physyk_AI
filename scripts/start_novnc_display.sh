#!/bin/bash
# ==============================================================================
# Physyk AI — High-Performance Virtual Display & noVNC Web Streamer (Pure TCP)
# Bypasses Corporate VPN / GlobalProtect UDP Blacklisting
# ==============================================================================

set -e

DISPLAY_NUM=":1"
VNC_PORT="5900"
NOVNC_PORT="6080"
RESOLUTION="1920x1080x24"

echo "======================================================================"
echo "  [Physyk Display] Starting Hardware-Accelerated Virtual X11 Display"
echo "  [Physyk Display] Resolution: $RESOLUTION | Web Port: $NOVNC_PORT (TCP)"
echo "======================================================================"

# Clean up any previous display processes
pkill -9 -f "Xvfb $DISPLAY_NUM" 2>/dev/null || true
pkill -9 -f "x11vnc.*$VNC_PORT" 2>/dev/null || true
pkill -9 -f "websockify.*$NOVNC_PORT" 2>/dev/null || true
pkill -9 -f "openbox" 2>/dev/null || true
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true
sleep 1

# 1. Start Virtual Framebuffer X11 Server with GLX and extensions
echo "1. Starting Xvfb on display $DISPLAY_NUM..."
Xvfb $DISPLAY_NUM -screen 0 $RESOLUTION +extension GLX +extension RENDER +extension RANDR -noreset > /opt/dlami/nvme/physyk/logs/xvfb.log 2>&1 &
sleep 2

# Export display environment for all child processes
export DISPLAY=$DISPLAY_NUM
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all

# 2. Start lightweight window manager
echo "2. Starting Openbox window manager..."
openbox > /opt/dlami/nvme/physyk/logs/openbox.log 2>&1 &
sleep 1

# 3. Start high-performance x11vnc server (TCP)
echo "3. Starting x11vnc on port $VNC_PORT..."
x11vnc -display $DISPLAY_NUM -nopw -forever -shared -rfbport $VNC_PORT -noxdamage -repeat -wait 5 -defer 5 > /opt/dlami/nvme/physyk/logs/x11vnc.log 2>&1 &
sleep 1

# 4. Start websockify + noVNC web client (TCP)
echo "4. Starting noVNC WebSocket bridge on port $NOVNC_PORT (TCP)..."
NOVNC_DIR="/usr/share/novnc"
if [ ! -d "$NOVNC_DIR" ]; then
    NOVNC_DIR="/usr/share/novnc-core"
fi

websockify --web $NOVNC_DIR $NOVNC_PORT localhost:$VNC_PORT > /opt/dlami/nvme/physyk/logs/websockify.log 2>&1 &
sleep 1

echo "======================================================================"
echo "  ✅ Virtual Display Active on $DISPLAY_NUM"
echo "  ✅ noVNC Web Interface running at: http://localhost:$NOVNC_PORT/vnc.html"
echo "======================================================================"
