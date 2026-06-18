from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensor

def _build_height_encoder(height_dim: int, hidden_dims: Sequence[int], latent_dim: int) -> nn.Sequential:
    """Build MLP encoder: ``height_dim → hidden_dims → latent_dim``."""
    layers: list[nn.Module] = []
    in_dim = height_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU()])
        in_dim = hidden_dim
    layers.extend([nn.Linear(in_dim, latent_dim), nn.ELU()])
    return nn.Sequential(*layers)

class height_scan_encoded(ManagerTermBase):
    """Height scanner raw scan을 MLP encoder latent로 변환하는 observation term.

    Isaac Lab ``image_features`` 와 동일하게 :class:`ManagerTermBase` 를 상속한다.
    ``height_scanner`` RayCaster 의 raw hit 높이를 읽은 뒤 encoder를 통과시켜
    ``latent_dim`` 차원 벡터를 policy observation 으로 제공한다.

    Encoder architecture (default)::

        Linear(height_dim, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 64) → ELU
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("height_scanner"))
        if isinstance(sensor_cfg, dict):
            sensor_cfg = SceneEntityCfg(**sensor_cfg)
        self._sensor_cfg: SceneEntityCfg = sensor_cfg

        self._offset: float = cfg.params.get("offset", 0.5)
        self._clip: tuple[float, float] | None = cfg.params.get("clip", (-1.0, 1.0))
        latent_dim: int = cfg.params.get("latent_dim", 64)
        hidden_dims: tuple[int, ...] = tuple(cfg.params.get("hidden_dims", (256, 128)))

        height_dim = env.scene.sensors[sensor_cfg.name].num_rays
        self.height_encoder = _build_height_encoder(height_dim, hidden_dims, latent_dim)
        self.height_encoder.to(env.device)

    @property
    def latent_dim(self) -> int:
        """Encoder output dimension."""
        for module in reversed(self.height_encoder):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Could not infer latent dimension from height encoder.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
        offset: float = 0.5,
        clip: tuple[float, float] | None = (-1.0, 1.0),
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> torch.Tensor:
        del env, sensor_cfg, offset, clip, latent_dim, hidden_dims  # resolved at init from cfg.params

        from isaaclab.envs.mdp.observations import height_scan as height_scan_raw

        raw_scan = height_scan_raw(self._env, self._sensor_cfg, offset=self._offset)
        if self._clip is not None:
            raw_scan = raw_scan.clip(min=self._clip[0], max=self._clip[1])
        return self.height_encoder(raw_scan)


def body_contact_forces_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Selected body contact forces in the robot base frame [N], concatenated per env.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor and body selection.
        asset_cfg: Robot articulation used for the base-frame rotation.

    Returns:
        Contact forces with shape ``(num_envs, 3 * num_bodies)`` as ``[fx, fy, fz]`` per body.
    """
    from isaaclab.sensors import ContactSensor
    from isaaclab.utils.math import quat_apply_inverse

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    robot: Articulation = env.scene[asset_cfg.name]
    root_quat_w = robot.data.root_quat_w.torch
    forces_b = quat_apply_inverse(root_quat_w.unsqueeze(1).expand(-1, forces_w.shape[1], -1), forces_w)
    return forces_b.reshape(env.num_envs, -1)


def relative_goal_position_b(
    env: ManagerBasedEnv,
    command_name: str,
    include_z: bool = False,
) -> torch.Tensor:
    """base-frame에서 본 목표의 상대 위치(원래 ``pos_command_b``).

    ``get_command`` 의 xy 는 단위 방향으로 노출될 수 있으므로, 거리/위치는 커맨드 텀 내부
    ``pos_command_b`` 를 직접 읽는다.

    Args:
        include_z: True 면 ``[Δx_b, Δy_b, Δz_b]`` (3D), False 면 ``[Δx_b, Δy_b]`` (2D) 반환.
    """
    cmd_term = env.command_manager.get_term(command_name)
    pos_b = cmd_term.pos_command_b
    if include_z:
        return pos_b[:, :3]
    return pos_b[:, :2]