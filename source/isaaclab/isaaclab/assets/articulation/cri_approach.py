"""Direction correction for the highest-CRI point. Joint speed is not scaled.

The library exports ``impact_jacobian`` and ``impact_dir``. The point with
the largest CRI supplies ``J`` and ``dir``:

    v_lin = J[..., :3, :] @ qd
    approach = (dir * v_lin).sum(-1)
    v_lin_tangential = v_lin - approach * dir

A 9-channel peak at or above ``point_limit`` applies
``min(1, direction_gain * peak / point_limit)`` of the joint change that
removes that point's incoming approach. Tangential speed stays, including a
peak already over 0.96. The uniform speed scale runs only when
``apply_speed_scale`` is set.
"""

from __future__ import annotations

import numpy as np
import torch

# Same defaults as the RLinf fridge filter. Speed scaling stays off.
CRI_HARD_LIMIT = 0.96
CRI_POINT_LIMIT = 0.92
CRI_DIRECTION_GAIN = 0.8
CRI_STRONG_CRI = 0.92
CRI_STRONG_TARGET = 0.0095


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
    """Remove the highest point's incoming approach. Do not scale joint speed.

    Args:
        qd: Requested joint velocity, ``(N, joints)``.
        jacobian: Per-point spatial Jacobian, ``(N, P, 6, joints)``.
        direction: Impact direction, ``(N, P, 3)``.
        cri: Per-point CRI, ``(N, C)``. The row peak uses every channel.
        hard_limit: Ceiling used only when the speed scale is on.
        point_limit: The peak point starts losing approach at or above this.
        strong_cri: 9-channel peak where the speed scale also applies.
        strong_target: CRI the speed scale aims at once ``strong_cri`` is hit.
        apply_speed_scale: When false, only the approach direction is changed.
        direction_gain: Weight of the approach-removal vector at ``point_limit``.
        qd_limit: Optional per-joint speed bound. Omitted leaves the direction
            change unscaled.

    Returns:
        Joint velocity with that point's incoming approach partly removed.
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
    peak_idx = point_cri.argmax(dim=-1)
    batch = torch.arange(n, device=out.device)
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3][batch, peak_idx]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    v_lin = torch.matmul(j_lin[batch, peak_idx], qd_j.unsqueeze(-1))
    approach = (impact_dir * v_lin.squeeze(-1)).sum(-1, keepdim=True)
    v_lin_tangential = v_lin.squeeze(-1) - approach * impact_dir
    entering = approach.squeeze(-1) > 0.0
    finite = torch.isfinite(v_lin_tangential).all(dim=-1)
    point_hot = point_cri[batch, peak_idx] >= float(point_limit)
    if not speed_on:
        point_hot = peak >= float(point_limit)
    drop_approach = (strong | (mild & point_hot)) & entering & finite
    delta_v = (v_lin_tangential - v_lin.squeeze(-1)).unsqueeze(-1)
    delta_v = torch.where(drop_approach.view(n, 1, 1), delta_v, torch.zeros_like(delta_v))
    dq = torch.linalg.pinv(j_lin[batch, peak_idx]) @ delta_v
    dq = dq.squeeze(-1)
    if speed_on:
        qd_new = qd_j + dq
    else:
        gain = (float(direction_gain) * peak / max(float(point_limit), 1e-6)).clamp(max=1.0)
        qd_new = qd_j + dq * gain.unsqueeze(-1)
    if speed_on:
        speed_scale = (float(strong_target) / peak.clamp_min(1e-6)).clamp(max=1.0)
        qd_new = torch.where(strong.unsqueeze(-1), qd_new * speed_scale.unsqueeze(-1), qd_new)
    if qd_limit is not None:
        lim = torch.as_tensor(qd_limit, device=out.device, dtype=out.dtype).reshape(-1)
        qd_new = qd_new.clamp(-lim[:joints], lim[:joints])
    use = drop_approach | strong
    out[:n, :joints] = torch.where(use.unsqueeze(-1), qd_new, qd_j)
    return out
