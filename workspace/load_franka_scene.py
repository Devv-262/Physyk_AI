#!/usr/bin/env python3
"""
Franka Panda scene loader — runs via --exec inside runheadless.webrtc.sh
Loads the Franka Panda USD + cameras + target objects into the live streaming viewport.
"""
import asyncio
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf, Sdf
import carb

async def load_franka_scene():
    print("\n[PHYSYK] ── Loading Franka Panda scene ──────────────────────────────")

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        await ctx.new_stage_async()
        stage = ctx.get_stage()

    # ── Nucleus assets root ─────────────────────────────────────────────────────
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("[PHYSYK] Cannot reach Nucleus — check container network")
        return
    print(f"[PHYSYK] Nucleus: {assets_root}")

    # ── Clear stage & set up units ──────────────────────────────────────────────
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    # ── Lighting ────────────────────────────────────────────────────────────────
    dome = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
    dome.GetAttribute("intensity").Set(1000.0)

    key = stage.DefinePrim("/World/Lights/KeyLight", "DistantLight")
    key.GetAttribute("intensity").Set(3000.0)
    key.GetAttribute("color").Set(Gf.Vec3f(1.0, 0.95, 0.85))
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

    fill = stage.DefinePrim("/World/Lights/FillLight", "SphereLight")
    fill.GetAttribute("intensity").Set(600.0)
    fill.GetAttribute("radius").Set(0.3)
    fill.GetAttribute("color").Set(Gf.Vec3f(0.55, 0.65, 1.0))
    UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(-1.5, 1.5, 2.5))

    # ── Ground plane ────────────────────────────────────────────────────────────
    ground = stage.DefinePrim("/World/GroundPlane", "Mesh")
    from pxr import UsdPhysics
    UsdPhysics.CollisionAPI.Apply(ground)

    # ── Franka Panda USD ────────────────────────────────────────────────────────
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    print(f"[PHYSYK] Loading: {franka_usd}")
    from omni.isaac.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
    print("[PHYSYK] Franka USD referenced ✓")

    # ── Red target cube ─────────────────────────────────────────────────────────
    cube = stage.DefinePrim("/World/TargetCube", "Cube")
    cube.GetAttribute("size").Set(0.05)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0.48, 0.0, 0.025))
    from pxr import UsdShade, Sdf
    mat_prim = stage.DefinePrim("/World/Materials/RedMat", "Material")
    shader = stage.DefinePrim("/World/Materials/RedMat/Shader", "Shader")
    shader.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    shader.CreateAttribute("inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.88, 0.15, 0.10))
    shader.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateAttribute("inputs:metallic", Sdf.ValueTypeNames.Float).Set(0.1)

    print("[PHYSYK] Scene objects added ✓")

    # ── Scene Camera ────────────────────────────────────────────────────────────
    scene_cam = stage.DefinePrim("/World/Cameras/SceneCamera", "Camera")
    scene_cam.GetAttribute("focalLength").Set(28.0)
    scene_cam.GetAttribute("horizontalAperture").Set(36.0)
    scene_cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 50.0))
    sc_xf = UsdGeom.Xformable(scene_cam)
    sc_xf.AddTranslateOp().Set(Gf.Vec3d(-1.1, -1.1, 1.5))
    sc_xf.AddRotateXYZOp().Set(Gf.Vec3f(-32, 0, -45))

    # ── Wrist Camera on panda_link8 ─────────────────────────────────────────────
    # NOTE: panda_link8 is the EE link — camera will move with the arm
    wrist_cam = stage.DefinePrim("/World/Franka/panda_link8/WristCamera", "Camera")
    wrist_cam.GetAttribute("focalLength").Set(10.0)
    wrist_cam.GetAttribute("horizontalAperture").Set(20.0)
    wrist_cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.005, 4.0))
    wc_xf = UsdGeom.Xformable(wrist_cam)
    wc_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.06))
    wc_xf.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

    print("[PHYSYK] Cameras configured:")
    print("[PHYSYK]   Scene camera → /World/Cameras/SceneCamera")
    print("[PHYSYK]   Wrist camera → /World/Franka/panda_link8/WristCamera")
    print("[PHYSYK] ─────────────────────────────────────────────────────────────")
    print("[PHYSYK] ✅ Scene ready. Switch viewport to SceneCamera to see the arm.")
    print("[PHYSYK] ✅ WebRTC stream is LIVE on port 8211")
    print("[PHYSYK] ─────────────────────────────────────────────────────────────\n")

# Schedule on the Kit event loop
asyncio.ensure_future(load_franka_scene())
