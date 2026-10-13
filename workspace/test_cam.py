import sys, cv2, time
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

from pxr import UsdLux, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep

world = World()
stage = world.stage
world.scene.add_default_ground_plane()

dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome_light.CreateIntensityAttr(800.0)

scene_cam = rep.create.camera(position=(1.2, -1.2, 0.9), look_at=(0.3, 0.0, 0.1))
rp_scene = rep.create.render_product(scene_cam, (640, 360))
scene_annot = rep.AnnotatorRegistry.get_annotator("rgb")
scene_annot.attach([rp_scene])

world.reset()
for _ in range(20):
    world.step(render=True)
    rep.orchestrator.step()

data = scene_annot.get_data()
if data is not None:
    img = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite("/tmp/test_cam_out.jpg", img)
    print("SAVED", img.mean())
else:
    print("NO DATA")
sim_app.close()
