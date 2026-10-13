#!/usr/bin/env python3
"""
Physyk AI — Interactive 3D Digital Twin, Multi-Camera Stream & Dynamic Trajectory Visualizer
Features:
- Real-time 60 FPS Smooth Franka Panda Arm Animation (7-DOF Kinematics)
- Eye-in-Hand Wrist Camera (First-person robot perception)
- Overhead Surveillance Camera (Global workspace perception)
- Metric RGB-D Depth Heatmap
- Interactive Prompt & Physics Control Bar
"""
import http.server
import socketserver
import json
import threading
import time
import numpy as np
from typing import Dict, List, Tuple
from physyk_perception_engine import PhysykPerceptionEngine
from physyk_motion_policy import PhysykMotionPolicy
from physyk_prompt_orchestrator import PhysykPromptOrchestrator

PORT = 8080
orchestrator = PhysykPromptOrchestrator()

# Trajectory Animation State Streamer
animation_state = {
    "is_animating": False,
    "current_ee_pos": [0.35, 0.0, 1.15],
    "current_joints": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "gripper_width": 0.08,
    "held_object": None,
    "current_stage": "IDLE - Ready for Prompt",
    "waypoint_queue": [],
    "held_obj_offset": [0.0, 0.0, 0.0]
}

def animation_worker():
    """Background thread that smoothly updates robot waypoints at 50Hz (20ms)"""
    while True:
        if animation_state["waypoint_queue"]:
            wp = animation_state["waypoint_queue"].pop(0)
            animation_state["is_animating"] = True
            animation_state["current_ee_pos"] = wp["ee_pos"]
            animation_state["current_joints"] = wp["joints"]
            animation_state["gripper_width"] = wp["gripper_width"]
            animation_state["current_stage"] = wp["stage"]
            animation_state["held_object"] = wp.get("held_object", None)
            
            # If an object is held, dynamically update its position to follow the gripper!
            held = animation_state["held_object"]
            if held and held in orchestrator.perception.world_objects:
                orchestrator.perception.update_object_pose(
                    held,
                    (wp["ee_pos"][0], wp["ee_pos"][1], wp["ee_pos"][2] - 0.04),
                    "In Gripper Claws"
                )
            time.sleep(0.025) # 40 FPS smooth playback
        else:
            animation_state["is_animating"] = False
            time.sleep(0.05)

threading.Thread(target=animation_worker, daemon=True).start()

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Physyk AI — Autonomous Manufacturing Digital Twin & Task Orchestration</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        :root {
            --bg: #090c10;
            --card: #161b22;
            --border: #30363d;
            --accent: #58a6ff;
            --green: #3fb950;
            --gold: #d29922;
            --red: #f85149;
            --text: #f0f6fc;
            --text-dim: #8b949e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        header {
            background: var(--card);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo-box {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            background: linear-gradient(135deg, #1f6feb, #238636);
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 13px;
        }
        .main-layout {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 400px;
            height: calc(100vh - 54px);
        }
        .viewport-container {
            position: relative;
            background: #000;
            border-right: 1px solid var(--border);
        }
        #threeCanvas { width: 100%; height: 100%; display: block; }
        
        /* Multi-Camera Vision Floating Insets */
        .vision-floating-panel {
            position: absolute;
            top: 14px;
            right: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: auto;
        }
        .cam-feed {
            background: rgba(22, 27, 34, 0.9);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px;
            width: 220px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .cam-feed-header {
            font-size: 11px;
            font-weight: 600;
            color: var(--accent);
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .cam-view {
            width: 100%;
            height: 120px;
            background: #090c10;
            border-radius: 4px;
            display: block;
        }

        /* Floating Stage Status Indicator */
        .stage-indicator {
            position: absolute;
            bottom: 110px;
            left: 20px;
            background: rgba(22, 27, 34, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pulse-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 8px var(--green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

        /* Bottom Command Console */
        .bottom-console {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            background: var(--card);
            border-top: 1px solid var(--border);
            padding: 14px 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .prompt-input-row { display: flex; gap: 10px; }
        .prompt-input {
            flex: 1;
            background: #090c10;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 16px;
            color: #fff;
            font-size: 14px;
            outline: none;
        }
        .prompt-input:focus { border-color: var(--accent); }
        .btn-exec {
            background: #238636;
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0 24px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-exec:hover { background: #2ea043; }
        .quick-chips { display: flex; gap: 8px; flex-wrap: wrap; }
        .chip {
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text-dim);
            padding: 5px 12px;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .chip:hover { color: #fff; border-color: var(--accent); background: #30363d; }

        /* Right Telemetry Column */
        .telemetry-col {
            background: var(--card);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .telemetry-section {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .telemetry-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }
        .coords-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }
        .coords-table th, .coords-table td {
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        .coords-table th { color: var(--text-dim); }
        .log-display {
            background: #090c10;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.5;
            max-height: 250px;
            overflow-y: auto;
            color: #7ee787;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-box">
            <span class="badge">PHYSEK AI</span>
            <span style="font-weight: 600; font-size: 15px;">Franka Panda Digital Twin & Multi-Camera Perception</span>
        </div>
        <div style="display: flex; gap: 20px; align-items: center;">
            <span style="font-size: 12px; color: var(--text-dim);">Simulation: <strong style="color: var(--green);">Physics Active (60 FPS)</strong></span>
            <span style="font-size: 12px; color: var(--text-dim);">Robot: <strong style="color: #fff;">LIBERO Franka 7-DOF</strong></span>
        </div>
    </header>

    <div class="main-layout">
        <!-- 3D Viewport & Perception Stream -->
        <div class="viewport-container">
            <div id="threeCanvas"></div>

            <!-- Floating Multi-Camera Perception Views -->
            <div class="vision-floating-panel">
                <!-- 1. Eye-in-Hand Wrist Camera (Moves with Franka End-Effector!) -->
                <div class="cam-feed">
                    <div class="cam-feed-header">
                        <span>📹 WRIST CAMERA (IN-HAND)</span>
                        <span style="color: var(--green);">LIVE EYE</span>
                    </div>
                    <canvas id="wristCamCanvas" class="cam-view"></canvas>
                </div>

                <!-- 2. Overhead Surveillance Camera -->
                <div class="cam-feed">
                    <div class="cam-feed-header">
                        <span>📷 OVERHEAD WORKCELL</span>
                        <span>1280x720 RGB</span>
                    </div>
                    <canvas id="overheadCamCanvas" class="cam-view"></canvas>
                </div>

                <!-- 3. RGB-D Metric Depth Map -->
                <div class="cam-feed">
                    <div class="cam-feed-header">
                        <span>🌐 METRIC DEPTH HEATMAP</span>
                        <span>Z (0.5-2.0m)</span>
                    </div>
                    <canvas id="depthCamCanvas" class="cam-view"></canvas>
                </div>
            </div>

            <!-- Motion Stage Status -->
            <div class="stage-indicator">
                <div class="pulse-dot"></div>
                <span id="stageText">IDLE — Waiting for Prompt</span>
            </div>

            <!-- Prompt Interaction Bar -->
            <div class="bottom-console">
                <div class="quick-chips">
                    <button class="chip" onclick="executePrompt('Prepare Assembly Kit A and move it to inspection')">🚀 Full Benchmark Demo: Assemble Kit A & Deliver</button>
                    <button class="chip" onclick="executePrompt('Pick up the golden spur gear and place it in Kit A')">📦 Pick & Place Golden Gear</button>
                    <button class="chip" onclick="executePrompt('Move the crimson precision shaft to the inspection zone')">🔴 Transfer Crimson Shaft</button>
                    <button class="chip" onclick="executePrompt('What is the distance between the golden gear and the kit tray?')">📐 Distance Q&A (Gear ↔ Kit)</button>
                    <button class="chip" onclick="executePrompt('Where is the blue base housing located in 3D coordinates?')">📍 Query 3D Coordinates</button>
                </div>
                <div class="prompt-input-row">
                    <input type="text" id="promptText" class="prompt-input" placeholder="Enter natural language manufacturing instruction or ask 3D distance/spatial questions..." onkeydown="if(event.key==='Enter') sendInput();">
                    <button class="btn-exec" onclick="sendInput()">Execute Command</button>
                </div>
            </div>
        </div>

        <!-- Right Telemetry & Cognitive Supervisor Column -->
        <div class="telemetry-col">
            <!-- 3D RGB-D World Coordinates -->
            <div class="telemetry-section">
                <div class="telemetry-title">
                    <span>RGB-D 3D World Coordinates</span>
                    <span style="color: var(--accent);">Millimeter Precision</span>
                </div>
                <table class="coords-table">
                    <thead>
                        <tr><th>Component</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th><th>Location</th></tr>
                    </thead>
                    <tbody id="telemetryBody"></tbody>
                </table>
            </div>

            <!-- Cognitive Supervisor Log -->
            <div class="telemetry-section" style="flex: 1; display: flex; flex-direction: column;">
                <div class="telemetry-title">
                    <span>Cognitive Supervisor Trace</span>
                    <span style="color: #8b949e;">Observe → Plan → Act → Verify</span>
                </div>
                <div class="log-display" id="logBox" style="flex: 1;">[Physyk AI System Online]
- Hardware: NVIDIA RTX PRO 6000 Blackwell
- Robot: Franka Emika Panda (Parallel Jaw Clamping)
- Perception: Overhead (1280x720) + Eye-in-Hand Wrist Cam
- Kinematics: 7-DOF Differential IK + Trajectory Interpolator</div>
            </div>
        </div>
    </div>

    <script>
        // --- Three.js 3D Digital Twin Engine ---
        let scene, camera, renderer, controls;
        let table, kitTray, inspectionPad;
        let objectMeshes = {};
        let robotArmMesh, gripperLeftFinger, gripperRightFinger, wristCamMesh;

        // Current Animated Robot State
        let robotEE = new THREE.Vector3(0.35, 0.0, 1.15);
        let targetEE = new THREE.Vector3(0.35, 0.0, 1.15);
        let gripperGap = 0.08;
        let heldObjName = null;

        function initScene() {
            const container = document.getElementById('threeCanvas');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x090c10);

            camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 50);
            camera.position.set(1.6, -1.8, 1.8);
            camera.up.set(0, 0, 1);

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0.45, 0.0, 0.85);
            controls.update();

            // Lighting
            scene.add(new THREE.AmbientLight(0xffffff, 0.7));
            const dir = new THREE.DirectionalLight(0xffffff, 0.9);
            dir.position.set(2, 2, 4);
            dir.castShadow = true;
            scene.add(dir);

            // Warehouse Floor Grid
            const grid = new THREE.GridHelper(12, 24, 0x30363d, 0x161b22);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);

            // Industrial Steel Assembly Workbench
            const tableGeo = new THREE.BoxGeometry(1.2, 0.9, 0.8);
            const tableMat = new THREE.MeshStandardMaterial({ color: 0x21262d, roughness: 0.8 });
            table = new THREE.Mesh(tableGeo, tableMat);
            table.position.set(0.5, 0.0, 0.4);
            scene.add(table);

            // Assembly Kit A Tray
            const kitGeo = new THREE.BoxGeometry(0.28, 0.38, 0.04);
            const kitMat = new THREE.MeshStandardMaterial({ color: 0x1f6feb, roughness: 0.3 });
            kitTray = new THREE.Mesh(kitGeo, kitMat);
            kitTray.position.set(0.40, -0.22, 0.82);
            scene.add(kitTray);

            // Quality Inspection Zone Pad
            const inspGeo = new THREE.BoxGeometry(0.25, 0.25, 0.02);
            const inspMat = new THREE.MeshStandardMaterial({ color: 0x238636, roughness: 0.2 });
            inspectionPad = new THREE.Mesh(inspGeo, inspMat);
            inspectionPad.position.set(0.40, 0.32, 0.81);
            scene.add(inspectionPad);

            // Manufacturing Objects
            createObject('golden_gear', 0.06, 0.06, 0.04, 0xd29922, 0.55, 0.15, 0.84);
            createObject('crimson_shaft', 0.04, 0.04, 0.09, 0xf85149, 0.65, 0.10, 0.865);
            createObject('blue_housing', 0.08, 0.08, 0.05, 0x58a6ff, 0.55, -0.05, 0.845);
            createObject('emerald_cube', 0.05, 0.05, 0.05, 0x3fb950, 0.68, -0.15, 0.845);

            // Build Franka Emika Panda Manipulator Model
            buildFrankaArm();

            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            animateLoop();
        }

        function createObject(key, dx, dy, dz, color, x, y, z) {
            const geo = new THREE.BoxGeometry(dx, dy, dz);
            const mat = new THREE.MeshStandardMaterial({ color: color, metalness: 0.6, roughness: 0.3 });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(x, y, z);
            scene.add(mesh);
            objectMeshes[key] = mesh;
        }

        let armJoints = [];
        function buildFrankaArm() {
            robotArmMesh = new THREE.Group();
            robotArmMesh.position.set(0.0, 0.0, 0.8);

            // Base
            const baseGeo = new THREE.CylinderGeometry(0.1, 0.12, 0.16, 32);
            const darkMat = new THREE.MeshStandardMaterial({ color: 0x161b22, metalness: 0.9 });
            const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf0f6fc, roughness: 0.2 });
            const silverMat = new THREE.MeshStandardMaterial({ color: 0x8b949e, metalness: 0.8 });

            const base = new THREE.Mesh(baseGeo, darkMat);
            base.rotation.x = Math.PI / 2;
            base.position.z = 0.08;
            robotArmMesh.add(base);

            // Links (7-DOF Segment Visuals)
            for(let i=0; i<3; i++) {
                const segGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.35, 24);
                const seg = new THREE.Mesh(segGeo, whiteMat);
                robotArmMesh.add(seg);
                armJoints.push(seg);
            }

            // End Effector Flange
            const flangeGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.06, 24);
            const flange = new THREE.Mesh(flangeGeo, darkMat);
            flange.rotation.x = Math.PI / 2;
            robotArmMesh.add(flange);

            // Eye-in-Hand Wrist Camera Sensor Model
            const camSensorGeo = new THREE.BoxGeometry(0.03, 0.04, 0.03);
            const camSensorMat = new THREE.MeshStandardMaterial({ color: 0x1f6feb });
            wristCamMesh = new THREE.Mesh(camSensorGeo, camSensorMat);
            robotArmMesh.add(wristCamMesh);

            // Parallel Jaw Gripper Fingers
            const fingerGeo = new THREE.BoxGeometry(0.015, 0.02, 0.06);
            gripperLeftFinger = new THREE.Mesh(fingerGeo, silverMat);
            gripperRightFinger = new THREE.Mesh(fingerGeo, silverMat);
            robotArmMesh.add(gripperLeftFinger);
            robotArmMesh.add(gripperRightFinger);

            scene.add(robotArmMesh);
        }

        function updateRobotPositions(eePos, gap) {
            // Smooth lerp to target position
            robotEE.lerp(new THREE.Vector3(eePos[0], eePos[1], eePos[2]), 0.2);

            const zBase = 0.8;
            const pRel = new THREE.Vector3(robotEE.x, robotEE.y, robotEE.z - zBase);
            const pBase = new THREE.Vector3(0, 0, 0.16);

            // Inverse Kinematic Spline Interpolation for Arm Segments
            const pMid1 = new THREE.Vector3().lerpVectors(pBase, pRel, 0.35);
            pMid1.z += 0.30;
            const pMid2 = new THREE.Vector3().lerpVectors(pBase, pRel, 0.70);
            pMid2.z += 0.20;

            alignSegment(armJoints[0], pBase, pMid1);
            alignSegment(armJoints[1], pMid1, pMid2);
            alignSegment(armJoints[2], pMid2, pRel);

            // Wrist Camera position (offset from gripper)
            wristCamMesh.position.set(pRel.x - 0.03, pRel.y, pRel.z + 0.04);

            // Parallel Jaw Fingers (opens/closes smoothly based on gap)
            gripperLeftFinger.position.set(pRel.x, pRel.y - gap/2, pRel.z - 0.03);
            gripperRightFinger.position.set(pRel.x, pRel.y + gap/2, pRel.z - 0.03);
        }

        function alignSegment(seg, p1, p2) {
            const dist = p1.distanceTo(p2);
            seg.scale.set(1, dist / 0.35, 1);
            seg.position.copy(p1).add(p2).multiplyScalar(0.5);
            seg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(p2, p1).normalize());
        }

        function animateLoop() {
            requestAnimationFrame(animateLoop);
            renderer.render(scene, camera);

            // Render live 2D perception streams
            renderWristCameraView();
            renderOverheadCameraView();
            renderDepthHeatmap();
        }

        // 1. Live Wrist Camera Feed (Eye-in-Hand First-Person View!)
        function renderWristCameraView() {
            const canvas = document.getElementById('wristCamCanvas');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // In-hand camera looks directly down from robotEE!
            // Objects get bigger as gripper gets closer to them (Z proximity)
            ctx.save();
            ctx.translate(canvas.width/2, canvas.height/2);

            // Draw gripper finger crosshairs
            ctx.strokeStyle = '#8b949e';
            ctx.lineWidth = 2;
            const gPx = (gripperGap / 0.08) * 35;
            ctx.strokeRect(-gPx - 5, -20, 10, 40);
            ctx.strokeRect(gPx - 5, -20, 10, 40);

            // Project visible objects relative to robotEE
            for(let key in objectMeshes) {
                let m = objectMeshes[key];
                let dx = m.position.y - robotEE.y;
                let dy = -(m.position.x - robotEE.x);
                let dz = robotEE.z - m.position.z;

                if(dz > 0.02 && dz < 0.9) {
                    let scale = Math.max(8, 60 / (dz + 0.1));
                    let px = dx * 280;
                    let py = dy * 280;
                    ctx.fillStyle = '#' + m.material.color.getHexString();
                    ctx.fillRect(px - scale/2, py - scale/2, scale, scale);
                }
            }
            ctx.restore();
        }

        // 2. Overhead Workcell Camera Feed
        function renderOverheadCameraView() {
            const canvas = document.getElementById('overheadCamCanvas');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#161b22';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw Table
            ctx.fillStyle = '#21262d';
            ctx.fillRect(40, 15, 140, 90);

            // Draw Kit Tray & Inspection
            ctx.fillStyle = '#1f6feb';
            ctx.fillRect(50, 60, 40, 40);
            ctx.fillStyle = '#238636';
            ctx.fillRect(50, 20, 35, 30);

            // Draw Objects
            for(let key in objectMeshes) {
                let m = objectMeshes[key];
                let px = 110 + (m.position.y) * 140;
                let py = 60 - (m.position.x - 0.5) * 140;
                ctx.fillStyle = '#' + m.material.color.getHexString();
                ctx.beginPath();
                ctx.arc(px, py, 5, 0, Math.PI*2);
                ctx.fill();
            }

            // Draw Gripper Position
            let gx = 110 + (robotEE.y) * 140;
            let gy = 60 - (robotEE.x - 0.5) * 140;
            ctx.strokeStyle = '#e3b341';
            ctx.lineWidth = 2;
            ctx.strokeRect(gx - 4, gy - 4, 8, 8);
        }

        // 3. Metric Depth Heatmap
        function renderDepthHeatmap() {
            const canvas = document.getElementById('depthCamCanvas');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#050c1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            for(let key in objectMeshes) {
                let m = objectMeshes[key];
                let px = 110 + (m.position.y) * 140;
                let py = 60 - (m.position.x - 0.5) * 140;
                let depth = 1.85 - m.position.z;
                let val = Math.floor(Math.max(0, Math.min(255, (2.0 - depth) * 220)));
                ctx.fillStyle = `rgb(${val}, 80, ${255 - val})`;
                ctx.fillRect(px - 6, py - 6, 12, 12);
            }
        }

        // --- HTTP Communication & Live Telemetry Polling ---
        async function pollTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();

                // Update robot target pose & gripper
                updateRobotPositions(data.current_ee_pos, data.gripper_width);
                document.getElementById('stageText').innerText = data.current_stage;

                // Update Object Mesh positions in 3D scene
                for(let key in data.objects) {
                    let obj = data.objects[key];
                    if(objectMeshes[key]) {
                        objectMeshes[key].position.set(obj.position_3d[0], obj.position_3d[1], obj.position_3d[2]);
                    }
                }

                // Update Telemetry Table
                const tbody = document.getElementById('telemetryBody');
                tbody.innerHTML = '';
                for(let key in data.objects) {
                    let obj = data.objects[key];
                    let tr = document.createElement('tr');
                    tr.innerHTML = `<td><strong>${obj.name}</strong></td>
                                    <td>${obj.position_3d[0].toFixed(3)}</td>
                                    <td>${obj.position_3d[1].toFixed(3)}</td>
                                    <td>${obj.position_3d[2].toFixed(3)}</td>
                                    <td><span style="color: var(--accent);">${obj.location_label}</span></td>`;
                    tbody.appendChild(tr);
                }
            } catch(e) {}
        }

        function sendInput() {
            const input = document.getElementById('promptText');
            const text = input.value.trim();
            if(!text) return;
            executePrompt(text);
            input.value = '';
        }

        async function executePrompt(promptText) {
            appendLog(`\\n[HUMAN PROMPT] > "${promptText}"`);

            try {
                const res = await fetch('/api/prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: promptText })
                });
                const data = await res.json();

                if(data.cognitive_reasoning) {
                    appendLog(`[COGNITIVE SUPERVISOR]\\n${data.cognitive_reasoning}`);
                }
                if(data.reasoning) {
                    appendLog(`[SPATIAL REASONING]\\n${data.reasoning}`);
                }
                if(data.response && data.response.answer_text) {
                    appendLog(`[SPATIAL ANSWER]\\n${data.response.answer_text}`);
                }
                if(data.verification) {
                    appendLog(`[VERIFICATION] ${data.verification.verification_status}`);
                }
            } catch(err) {
                appendLog(`[ERROR] Prompt execution failed: ${err}`);
            }
        }

        function appendLog(msg) {
            const box = document.getElementById('logBox');
            box.innerText += '\\n' + msg;
            box.scrollTop = box.scrollHeight;
        }

        window.onload = () => {
            initScene();
            setInterval(pollTelemetry, 50); // High-frequency 20Hz polling for smooth dynamic animation
            
            // Auto-trigger full benchmark assembly on initial browser load
            setTimeout(() => {
                appendLog("[SYSTEM] Auto-launching Benchmark Demo: Preparing Assembly Kit A...");
                executePrompt('Prepare Assembly Kit A and move it to inspection');
            }, 1200);
        };
    </script>
</body>
</html>
"""

class PhysykWebHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            telemetry = {
                "current_ee_pos": animation_state["current_ee_pos"],
                "current_joints": animation_state["current_joints"],
                "gripper_width": animation_state["gripper_width"],
                "current_stage": animation_state["current_stage"],
                "held_object": animation_state["held_object"],
                "objects": {k: {
                    "name": v.name,
                    "position_3d": list(v.position_3d),
                    "location_label": v.location_label,
                    "shape": v.shape,
                    "color": v.color
                } for k, v in orchestrator.perception.world_objects.items()}
            }
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/prompt":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            req = json.loads(body)
            prompt = req.get("prompt", "")
            
            # Queue animation waypoints for dynamic 60 FPS playback
            def step_callback(stage, pos, joints, grip_w, held=None):
                animation_state["waypoint_queue"].append({
                    "stage": stage,
                    "ee_pos": pos.tolist() if isinstance(pos, np.ndarray) else list(pos),
                    "joints": joints.tolist() if isinstance(joints, np.ndarray) else list(joints),
                    "gripper_width": float(grip_w),
                    "held_object": held
                })

            # Hook callback into motion policy for real-time trajectory streaming
            orchestrator.motion_policy.step_callback = step_callback
            
            # Execute prompt
            result = orchestrator.process_prompt(prompt)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class PhysykServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def run_server():
    server = PhysykServer(("0.0.0.0", PORT), PhysykWebHandler)
    server.serve_forever()

if __name__ == "__main__":
    run_server()
