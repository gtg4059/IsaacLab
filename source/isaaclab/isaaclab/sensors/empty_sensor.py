from omni.isaac.lab.sim import SceneEntityCfg
from omni.isaac.lab.utils import configclass
import numpy as np
from dataclasses import MISSING

@configclass
class EmptyObjectCfg(SceneEntityCfg):
    prim_path: str = MISSING
    pose: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, qw, qx, qy, qz]

    def create(self, scene):
        from omni.isaac.core.prims import XFormPrim
        empty_prim = XFormPrim(self.prim_path, position=self.pose[:3], orientation=self.pose[3:])
        empty_prim.initialize()
        return empty_prim