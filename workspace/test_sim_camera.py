import time
import numpy as np
import cv2

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import GroundPlane, DistantLight, Cube
from isaacsim.sensors.camera import Camera

stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/DistantLight").set_intensities(600)
Cube("/World/TargetCube", positions=[0.45, 0.0, 0.05], scales=[0.08, 0.08, 0.08], colors="red")

camera = Camera(
    prim_path="/World/SceneCamera",
    position=np.array([1.2, -0.6, 0.9]),
    resolution=(640, 480)
)
camera.initialize()
app_utils.play()

for _ in range(15):
    app.update()

rgba = camera.get_rgba()
print("Camera RGBA shape:", rgba.shape if rgba is not None else None)
if rgba is not None and rgba.size > 0:
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite("/opt/dlami/nvme/physyk/workspace/test_camera_frame.jpg", bgr)
    print("SUCCESS: Camera captured frame, size =", rgba.shape)

app.close()
