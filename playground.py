from pathlib import Path
import numpy as np

import math
import numpy as np

from brainrender import Scene
from brainrender.actors import Cylinder
from brainrender import VideoMaker
from brainrender import Animation
from brainrender import settings
from bg_atlasapi import show_atlases
from brainrender.actors import Points

scene = Scene(title="Silicon Probe Visualization")

# Visualise the probe target regions
cp = scene.add_brain_region("CP", alpha=0.15)
rsp = scene.add_brain_region("RSP", alpha=0.15)
# render
scene.render()
print("")