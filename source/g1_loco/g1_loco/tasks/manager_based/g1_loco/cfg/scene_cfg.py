# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene 설정: 지형/로봇/조명/센서."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from g1_loco.utils.terrain import RANDOM_ROUGH_TERRAINS_CFG
from g1_loco.utils.unitree import G1_DEX_FIX


@configclass
class G1LocoSceneCfg(InteractiveSceneCfg):
    """G1 보행 학습용 장면(Scene) 구성.

    역할:
    - 지형(terrain) 생성 방식
    - 로봇 스폰 설정
    - 센서(높이 스캐너/접촉 센서)
    - 조명
    """

    # 지형 생성기 기반 바닥. 기본은 rough terrain generator를 사용한다.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RANDOM_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 학습에 사용할 G1 로봇 아티큘레이션 설정
    robot: ArticulationCfg = G1_DEX_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 장면 조명
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # 높이 스캐너: 바닥 높이 정보를 읽어 rough terrain 대응에 사용
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.8, 0.5]),
        max_distance=100.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 접촉 센서: 발 접촉/공중 시간 등 보상 계산에 사용
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
