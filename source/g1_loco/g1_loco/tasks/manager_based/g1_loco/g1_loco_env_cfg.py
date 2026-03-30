# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 보행 학습 환경의 기본 설정 파일.

역할별 파일로 분리된 설정을 조립하며,
기존 import 경로(`g1_loco_env_cfg`)와의 호환성을 유지한다.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .cfg.actions_cfg import ActionsCfg
from .cfg.commands_cfg import CommandsCfg
from .cfg.events_cfg import EventCfg
from .cfg.observations_cfg import ObservationsCfg
from .cfg.rewards_cfg import RewardsCfg
from .cfg.scene_cfg import G1LocoSceneCfg
from .cfg.terminations_cfg import TerminationsCfg


@configclass
class G1LocoEnvCfg(ManagerBasedRLEnvCfg):
    """G1 보행 환경의 공통 베이스 설정.

    구성 요소(scene/obs/actions/commands/events/rewards/terminations)를
    하나의 RL 환경 설정으로 조립한다.
    """

    # Scene settings
    scene: G1LocoSceneCfg = G1LocoSceneCfg(num_envs=2, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 보상/이벤트 계산에 필요한 접촉센서를 로봇 스폰 시 활성화
    scene.robot.spawn.activate_contact_sensors = True
    # 높이 스캐너 기준 프레임을 torso_link로 맞춤
    scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
    # 학습 시작 시 초기 지형 레벨을 제한해 안정적 초기 수렴 유도
    scene.terrain.max_init_terrain_level = 0

    def __post_init__(self) -> None:
        """환경 공통 런타임 파라미터 최종 설정."""
        # general settings
        self.decimation = 4  # 물리 업데이트 주기 설정 -> 200/4 = 50Hz
        self.episode_length_s = 20.0  # 에피소드 길이 설정

        # simulation settings
        self.sim.dt = 0.005  # 시뮬레이션 업데이트 주기 설정 -> 0.005s = 200Hz
        self.sim.render_interval = self.decimation  # 렌더링 주기 설정 -> 50Hz
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # terrain 커리큘럼 항목이 있으면 지형 생성기 커리큘럼을 켠다.
        # (파생 cfg에서 curriculum 유무에 따라 자동으로 on/off)
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1LocoEnvCfg_PLAY(G1LocoEnvCfg):
    """학습된 정책 확인(play)용 경량 설정."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # play에서는 커리큘럼을 끄고 고정 조건에서 동작 확인
        self.curriculum.terrain_levels = None

        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # spawn only on easiest terrain (minimum difficulty)
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.scene.terrain.max_init_terrain_level = None


__all__ = [
    "G1LocoSceneCfg",
    "CommandsCfg",
    "ActionsCfg",
    "ObservationsCfg",
    "EventCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "G1LocoEnvCfg",
    "G1LocoEnvCfg_PLAY",
]
