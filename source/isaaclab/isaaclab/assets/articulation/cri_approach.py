"""Slow the whole linear velocity of every hot collision point.

The library exports ``impact_jacobian`` and ``impact_dir``, one pair per
collision point. UR10's analysis file has 7 body points; the solver row also
includes the flange and the tool. A point at or above ``point_limit`` that is
moving into the obstacle keeps its linear direction and drops speed:

    v_lin = J[..., :3, :] @ qd
    gain = min(1, direction_gain * point_cri / point_limit)
    v_new = (1 - gain) * v_lin

``direction_gain`` 0.9 removes 90% of that whole vector at the point limit.
The gain uses the point CRI capped at 0.96, so the point target does not go to
zero. A command whose peak is at or above 0.96 is then multiplied by
``CRI_HARD_SCALE`` (0.1) once. The joint change is the one least-squares
solution of those cartesian targets. The uniform joint-speed scale runs only
when ``apply_speed_scale`` is set.
"""

from __future__ import annotations

import numpy as np
import torch

# Whole linear velocity is scaled. 90% is removed at the point limit.
# A peak at or above the hard limit is multiplied by this scale once.
CRI_HARD_LIMIT = 0.96
# Inward motion starts turning sideways at this point CRI. Speed stays.
CRI_VEER_AT = 0.7
CRI_HARD_SCALE = 0.1
# At this point CRI the inward collision-direction vector is cut to this fraction.
CRI_COLLISION_AT = 0.8
CRI_COLLISION_KEEP = 0.1
CRI_POINT_LIMIT = 0.92
CRI_DIRECTION_GAIN = 0.9
CRI_STRONG_CRI = 0.92
CRI_STRONG_TARGET = 0.0095


def keep_collision_component(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    cri_at: float = CRI_COLLISION_AT,
    keep: float = CRI_COLLISION_KEEP,
    hard_at: float = CRI_HARD_LIMIT,
) -> torch.Tensor:
    """Shrink the inward collision-direction vector. Tangential velocity stays.

    A point at or above ``cri_at`` and below ``hard_at`` keeps ``keep`` of
    ``(dir · v_lin) dir``. Points at or above ``hard_at`` are left unchanged.
    The joint change is the least-squares solution of those cartesian targets.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if cri_t.ndim == 3:
        cri_t = cri_t.amax(dim=0) if int(cri_t.shape[0]) != int(out.shape[0]) else cri_t.amax(dim=1)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    point_cri = cri_t[:n, :points]
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    approach = (impact_dir * v_lin).sum(dim=-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    entering = (approach > 0.0) & finite
    hot = (point_cri >= float(cri_at)) & (point_cri < float(hard_at)) & entering
    if not bool(hot.any()):
        return out
    drop = (1.0 - float(keep)) * approach
    delta_v = -drop.unsqueeze(-1) * impact_dir
    rhs = torch.where(hot.unsqueeze(-1), delta_v, torch.zeros_like(v_lin))
    j_used = torch.where(hot.view(n, points, 1, 1), j_lin, torch.zeros_like(j_lin))
    dq = torch.linalg.pinv(j_used.reshape(n, points * 3, joints)) @ rhs.reshape(n, points * 3, 1)
    qd_new = qd_j + dq.squeeze(-1)
    out[:n, :joints] = torch.where(hot.any(dim=-1).unsqueeze(-1), qd_new, qd_j)
    return out


def scale_over_joints(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    cri_at: float = CRI_HARD_LIMIT,
    share: float = 0.02,
) -> torch.Tensor:
    """Set the joints pushing into an over-limit point to zero speed.

    A joint is over when its share of the inward approach speed is at least
    ``share``. Those joint rates become 0. Joints that are not pushing in,
    including the ones that carry the tangential velocity, stay as they are.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if cri_t.ndim == 3:
        cri_t = cri_t.amax(dim=0) if int(cri_t.shape[0]) != int(out.shape[0]) else cri_t.amax(dim=1)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    elif int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    approach = (impact_dir * v_lin).sum(dim=-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    hot = (cri_t[:n, :points] >= float(cri_at)) & (approach > 0.0) & finite
    if not bool(hot.any()):
        return out
    axis = (j_lin * impact_dir.unsqueeze(-1)).sum(dim=-2)
    contrib = axis * qd_j[:, None, :]
    thresh = float(share) * approach.clamp_min(0.0)
    pushing = hot.unsqueeze(-1) & (contrib > thresh.unsqueeze(-1))
    over_joint = pushing.any(dim=1)
    qd_new = torch.where(over_joint, torch.zeros_like(qd_j), qd_j)
    out[:n, :joints] = qd_new
    return out


def veer_off_inward(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    cri_at: float = CRI_VEER_AT,
) -> torch.Tensor:
    """Turn inward velocity at a hot point sideways. The speed is kept.

    A point at or above ``cri_at`` whose motion still has an inward component
    along the impact direction is retargeted. The new linear velocity has the
    same speed and no inward part. An existing tangential part chooses the
    sideways direction. A purely inward velocity uses the impact direction
    crossed with world up. The joint change is the least-squares solution of
    that cartesian target. Joint rates are not zeroed.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if cri_t.ndim == 3:
        cri_t = cri_t.amax(dim=0) if int(cri_t.shape[0]) != int(out.shape[0]) else cri_t.amax(dim=1)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    elif int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    approach = (impact_dir * v_lin).sum(dim=-1)
    speed = v_lin.norm(dim=-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    inward = approach > torch.maximum(speed * 0.02, speed.new_full(speed.shape, 1.0e-3))
    hot = (cri_t[:n, :points] >= float(cri_at)) & inward & finite & (speed > 1.0e-4)
    if not bool(hot.any()):
        return out
    v_tang = v_lin - approach.unsqueeze(-1) * impact_dir
    tang_n = v_tang.norm(dim=-1)
    up = v_lin.new_tensor([0.0, 0.0, 1.0])
    side_z = torch.cross(impact_dir, up.expand_as(impact_dir), dim=-1)
    side_x = torch.cross(impact_dir, v_lin.new_tensor([1.0, 0.0, 0.0]).expand_as(impact_dir), dim=-1)
    parallel = side_z.norm(dim=-1) < 1.0e-4
    side = torch.where(parallel.unsqueeze(-1), side_x, side_z)
    side = side / side.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    use_tang = tang_n > torch.maximum(speed * 0.02, speed.new_full(speed.shape, 1.0e-3))
    heading = torch.where(use_tang.unsqueeze(-1), v_tang / tang_n.unsqueeze(-1).clamp_min(1e-8), side)
    v_new = heading * speed.unsqueeze(-1)
    rhs = torch.where(hot.unsqueeze(-1), v_new - v_lin, torch.zeros_like(v_lin))
    j_used = torch.where(hot.view(n, points, 1, 1), j_lin, torch.zeros_like(j_lin))
    dq = torch.linalg.pinv(j_used.reshape(n, points * 3, joints)) @ rhs.reshape(n, points * 3, 1)
    qd_new = qd_j + dq.squeeze(-1)
    out[:n, :joints] = torch.where(hot.any(dim=-1).unsqueeze(-1), qd_new, qd_j)
    return out


def zero_slide_joints(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    cri_at: float = CRI_HARD_LIMIT,
    share: float = 0.02,
) -> torch.Tensor:
    """Zero joints that push into a hot point or slide past it.

    A point at or above ``cri_at`` selects both sets. A pushing joint's share of
    the inward approach is at least ``share``. A sliding joint's share of the
    tangential linear speed is at least ``share``. Those rates become 0. Joint
    rates are never increased.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if cri_t.ndim == 3:
        cri_t = cri_t.amax(dim=0) if int(cri_t.shape[0]) != int(out.shape[0]) else cri_t.amax(dim=1)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    elif int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    approach = (impact_dir * v_lin).sum(dim=-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    hot = (cri_t[:n, :points] >= float(cri_at)) & finite
    if not bool(hot.any()):
        return out
    axis = (j_lin * impact_dir.unsqueeze(-1)).sum(dim=-2)
    inward = axis * qd_j[:, None, :]
    pushing = hot.unsqueeze(-1) & (approach.unsqueeze(-1) > 0.0) & (inward > float(share) * approach.clamp_min(0.0).unsqueeze(-1))
    j_tang = j_lin - axis.unsqueeze(-2) * impact_dir.unsqueeze(-1)
    tang_contrib = (j_tang * qd_j[:, None, None, :]).norm(dim=-2)
    v_tang = v_lin - approach.unsqueeze(-1) * impact_dir
    tang_speed = v_tang.norm(dim=-1)
    sliding = hot.unsqueeze(-1) & (tang_speed.unsqueeze(-1) > 1.0e-6) & (
        tang_contrib > float(share) * tang_speed.unsqueeze(-1)
    )
    stop = (pushing | sliding).any(dim=1)
    out[:n, :joints] = torch.where(stop, torch.zeros_like(qd_j), qd_j)
    return out


def zero_point_linear_velocity(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    cri_at: float = CRI_HARD_LIMIT,
) -> torch.Tensor:
    """Set the whole linear velocity to 0 at points whose CRI is at or above ``cri_at``."""
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if cri_t.ndim == 3:
        cri_t = cri_t.amax(dim=0) if int(cri_t.shape[0]) != int(out.shape[0]) else cri_t.amax(dim=1)
    if jac.ndim != 4 or jac.shape[-2] < 3:
        return out
    n = min(int(out.shape[0]), int(jac.shape[0]), int(cri_t.shape[0]))
    points = min(int(jac.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    j_lin = jac[:n, :points, :3, :joints]
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    hot = (cri_t[:n, :points] >= float(cri_at)) & finite
    if not bool(hot.any()):
        return out
    rhs = torch.where(hot.unsqueeze(-1), -v_lin, torch.zeros_like(v_lin))
    j_used = torch.where(hot.view(n, points, 1, 1), j_lin, torch.zeros_like(j_lin))
    dq = torch.linalg.pinv(j_used.reshape(n, points * 3, joints)) @ rhs.reshape(n, points * 3, 1)
    qd_new = qd_j + dq.squeeze(-1)
    out[:n, :joints] = torch.where(hot.any(dim=-1).unsqueeze(-1), qd_new, qd_j)
    return out


def subtract_approach_velocity(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    hard_limit: float = CRI_HARD_LIMIT,
    point_limit: float = CRI_POINT_LIMIT,
    strong_cri: float = CRI_STRONG_CRI,
    strong_target: float = CRI_STRONG_TARGET,
    apply_speed_scale: bool = False,
    direction_gain: float = CRI_DIRECTION_GAIN,
    qd_limit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slow the whole linear velocity at every point over ``point_limit``.

    Args:
        qd: Requested joint velocity, ``(N, joints)``.
        jacobian: Per-point spatial Jacobian, ``(N, P, 6, joints)``.
        direction: Impact direction, ``(N, P, 3)``.
        cri: Per-point CRI, ``(N, C)``. The row peak uses every channel.
        hard_limit: Ceiling used when the uniform speed scale is on. A peak at or
            above this is scaled once by ``CRI_HARD_SCALE`` in ``apply_cri_filter``.
        point_limit: A point starts slowing down at or above this.
        strong_cri: 9-channel peak where the speed scale also applies.
        strong_target: CRI the speed scale aims at once ``strong_cri`` is hit.
        apply_speed_scale: Extra uniform joint-speed scale after the point slowdown.
        direction_gain: Fraction of the whole linear velocity removed at
            ``point_limit``. The direction is unchanged.
        qd_limit: Optional per-joint speed bound. Omitted leaves the slowdown
            unclipped.

    Returns:
        Joint velocity whose hot points keep their linear direction at lower speed.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    peak = cri_t[:n].amax(dim=-1)
    speed_on = bool(apply_speed_scale)
    if speed_on:
        strong = peak >= float(strong_cri)
        mild = (peak < float(hard_limit)) & ~strong
        if not bool((strong | mild).any()):
            return out
    else:
        strong = torch.zeros(n, dtype=torch.bool, device=out.device)
        mild = torch.ones(n, dtype=torch.bool, device=out.device)
    point_cri = cri_t[:n, :points]
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    # v_lin is (N, P, 3). Every point over the limit is one block of rows.
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    approach = (impact_dir * v_lin).sum(dim=-1)
    entering = approach > 0.0
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    point_hot = point_cri >= float(point_limit)
    row_ok = (strong | mild) if speed_on else torch.ones(n, dtype=torch.bool, device=out.device)
    active = row_ok.unsqueeze(-1) & point_hot & entering & finite
    # Past the hard limit the extra reduction is one joint-speed scale of
    # CRI_HARD_SCALE. The point gain itself stays at the hard-limit value.
    cri_for_gain = point_cri.clamp(max=float(hard_limit))
    gain = (float(direction_gain) * cri_for_gain / max(float(point_limit), 1.0e-6)).clamp(0.0, 1.0)
    v_target = v_lin * (1.0 - gain).unsqueeze(-1)
    if speed_on:
        v_target = torch.where(strong.view(n, 1, 1), torch.zeros_like(v_target), v_target)
    # The whole linear velocity keeps its direction and drops ``gain``.
    rhs = torch.where(active.unsqueeze(-1), v_target - v_lin, torch.zeros_like(v_lin))
    j_used = torch.where(active.view(n, points, 1, 1), j_lin, torch.zeros_like(j_lin))
    dq = torch.linalg.pinv(j_used.reshape(n, points * 3, joints)) @ rhs.reshape(n, points * 3, 1)
    dq = dq.squeeze(-1)
    qd_new = qd_j + dq
    if speed_on:
        speed_scale = (float(strong_target) / peak.clamp_min(1e-6)).clamp(max=1.0)
        qd_new = torch.where(strong.unsqueeze(-1), qd_new * speed_scale.unsqueeze(-1), qd_new)
    if qd_limit is not None:
        lim = torch.as_tensor(qd_limit, device=out.device, dtype=out.dtype).reshape(-1)
        qd_new = qd_new.clamp(-lim[:joints], lim[:joints])
    use = active.any(dim=-1) | strong
    out[:n, :joints] = torch.where(use.unsqueeze(-1), qd_new, qd_j)
    return out


def scale_full_linear_velocity(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    limit: float = CRI_HARD_LIMIT,
    qd_limit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale the whole linear velocity of every point over ``limit``.

    The approach part and the sideways part keep their direction and share one
    factor, ``limit / point_cri``. A point at or under the limit is unchanged.
    The joint change is the least-squares solution of those cartesian targets.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if jac.ndim != 4 or jac.shape[-2] < 3:
        return out
    n = out.shape[0]
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    point_cri = cri_t[:n, :points]
    j_lin = jac[:n, :points, :3, :joints]
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin, qd_j[:, None, :, None]).squeeze(-1)
    finite = torch.isfinite(v_lin).all(dim=-1) & torch.isfinite(j_lin).flatten(-2).all(dim=-1)
    active = (point_cri > float(limit)) & finite
    if not bool(active.any()):
        return out
    keep = (float(limit) / point_cri.clamp_min(1.0e-6)).clamp(max=1.0)
    v_target = v_lin * keep.unsqueeze(-1)
    rhs = torch.where(active.unsqueeze(-1), v_target - v_lin, torch.zeros_like(v_lin))
    j_used = torch.where(active.view(n, points, 1, 1), j_lin, torch.zeros_like(j_lin))
    dq = torch.linalg.pinv(j_used.reshape(n, points * 3, joints)) @ rhs.reshape(n, points * 3, 1)
    qd_new = qd_j + dq.squeeze(-1)
    if qd_limit is not None:
        lim = torch.as_tensor(qd_limit, device=out.device, dtype=out.dtype).reshape(-1)
        qd_new = qd_new.clamp(-lim[:joints], lim[:joints])
    out[:n, :joints] = torch.where(active.any(dim=-1).unsqueeze(-1), qd_new, qd_j)
    return out


def scale_velocity_below_cri(
    qd: torch.Tensor,
    over: torch.Tensor,
    cri_peak_of,
    *,
    limit: float = CRI_HARD_LIMIT,
    iters: int = 6,
) -> torch.Tensor:
    """Shrink commands whose CRI is still at or above ``limit``.

    ``cri_peak_of(qd)`` returns the collision-point peak of that joint velocity.
    ``over`` marks rows whose unfiltered request already reached ``limit``.
    A row whose current command is already under the limit is kept. The other
    rows take the largest scale the callback reported under the limit. If every
    probe stays at or above it, that row is sent at zero speed.

    Matches the RLinf hard-limit search: six steps resolve the scale to 1/64.
    """
    if qd.numel() == 0 or not bool(over.any()):
        return qd
    peak = cri_peak_of(qd)
    still = over.reshape(-1)[: qd.shape[0]] & (peak >= float(limit))
    if not bool(still.any()):
        return qd
    lo = torch.zeros(qd.shape[0], device=qd.device, dtype=qd.dtype)
    hi = torch.ones_like(lo)
    best = torch.zeros_like(lo)
    for _ in range(int(iters)):
        mid = (lo + hi) * 0.5
        trial = qd.clone()
        trial[still] = qd[still] * mid[still].unsqueeze(-1)
        probed = cri_peak_of(trial)
        ok = still & (probed < float(limit))
        best = torch.where(ok, mid, best)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(still & ~ok, mid, hi)
    scaled = qd * best.unsqueeze(-1)
    return torch.where(still.unsqueeze(-1), scaled, qd)
