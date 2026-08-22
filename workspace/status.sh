#!/bin/bash
echo "════════════════════════════════════════════════════════════"
echo "  PHYSYK — System Status & Access Guide"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check Physyk server
if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
    echo "✅ PHYSYK FastAPI Server"
    echo "   Status: RUNNING (PID: $(pgrep -f 'physyk_main_server' | head -1))"
    echo "   Local:  http://localhost:7860"
    echo "   API:    http://localhost:7860/docs"
    echo ""
else
    echo "❌ PHYSYK FastAPI Server"
    echo "   Status: NOT RUNNING"
    echo ""
fi

# Check Isaac Sim
if docker ps | grep -q physyk-isaac-sim; then
    CONTAINER_ID=$(docker ps | grep physyk-isaac-sim | awk '{print $1}')
    echo "✅ Isaac Sim Container"
    echo "   Status: RUNNING (ID: $CONTAINER_ID)"
    echo ""
    echo "   ⚠️  WebRTC Streaming Status:"
    docker logs "$CONTAINER_ID" 2>&1 | grep -q "app ready" && echo "   Isaac Sim App: READY" || echo "   Isaac Sim App: STARTING"
    echo ""
    echo "   Try accessing WebRTC at:"
    echo "   • http://localhost:8211/streaming/webrtc-demo/"
    echo "   • http://localhost:8080"
    echo "   • Check docker logs: docker logs $CONTAINER_ID"
else
    echo "❌ Isaac Sim Container"
    echo "   Status: NOT RUNNING"
    echo "   Start with: bash /opt/dlami/nvme/physyk/scripts/launch_physyk_sim.sh webrtc"
    echo ""
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 HOW TO USE:"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "1. CONTROL THE ROBOT (Physyk Web UI):"
echo "   Open: http://localhost:7860"
echo "   Type commands like: 'pick up the cube', 'open gripper'"
echo ""
echo "2. WATCH THE SIMULATION (Isaac Sim):"
echo "   Open: http://localhost:8211/streaming/webrtc-demo/"
echo "   (If 8211 doesn't work, check Isaac Sim logs)"
echo ""
echo "3. PROGRAM CONTROL (API):"
echo "   POST to http://localhost:7860/execute"
echo '   Body: {"instruction": "your command"}'
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "🔗 All Listening Ports:"
ss -tuln | grep LISTEN | awk '{print $4}' | sort -u
echo ""
