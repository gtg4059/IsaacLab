# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video",
    dest="video",
    action="store_true",
    help="Record an rgb video under the checkpoint log directory (default: on).",
)
parser.add_argument(
    "--no_video",
    dest="video",
    action="store_false",
    help="Disable video recording.",
)
parser.set_defaults(video=True)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in env steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or os.environ.get("SFD_PLAY_CLIPS", "").strip():
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import reach_traj_csv_utils as csv_utils

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# CRI point order for UR10 + 2FG7: seven arm points, then the gripper point.
# Offsets are the analysis ColliPos, millimetres converted to metres, in that link frame.
# The gripper has no mesh of its own. It is a fixed joint on wrist_3.
_CRI_POINT_LINKS = (
    "upper_arm_link",
    "forearm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "wrist_3_link",
)
_CRI_POINT_OFFSET_M = (
    (0.0, 0.070, 0.0),
    (0.612, 0.0, 0.250),
    (0.0, 0.0, 0.240),
    (0.5723, 0.0, -0.010),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, -0.050),
    (0.0, 0.0, -0.050),
    (0.0, 0.0, 0.160),
)


class _CriLinkColorWrapper(gym.Wrapper):
    """Red sphere on each collision point whose CRI is at or above the mark.

    The UR10 meshes stay on a shared instance, so the recorded image uses a
    sphere instead of a material change. ``threshold`` defaults to the OVF
    line. Clip recordings pass 0.96.
    """

    def __init__(self, env, threshold: float | None = None):
        super().__init__(env)
        self._markers = None
        self._body_index: torch.Tensor | None = None
        self._offset: torch.Tensor | None = None
        self._threshold = 1.99 if threshold is None else float(threshold)
        if threshold is None:
            term = getattr(env.unwrapped, "termination_manager", None)
            names = getattr(term, "active_terms", None)
            if term is not None and names is not None and "OVF" in names:
                self._threshold = float(term.get_term_cfg("OVF").params["threshold"])

    def step(self, action):
        result = self.env.step(action)
        self._paint()
        return result

    def _paint(self) -> None:
        from isaaclab.utils import math as math_utils

        asset = self.unwrapped.scene["robot"]
        cri = asset.data.CRI.detach()
        pts = cri.reshape(cri.shape[0], -1)
        n = min(int(pts.shape[-1]), len(_CRI_POINT_LINKS))
        if self._body_index is None:
            names = list(asset.body_names)
            self._body_index = torch.tensor(
                [names.index(link) for link in _CRI_POINT_LINKS[:n]], device=asset.device, dtype=torch.long
            )
            self._offset = torch.tensor(_CRI_POINT_OFFSET_M[:n], device=asset.device, dtype=torch.float32)
        hot = pts[:, :n] >= self._threshold
        if self._markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/World/Visuals/cri_over",
                    markers={
                        "hot": sim_utils.SphereCfg(
                            radius=0.055,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.02, 0.02)),
                        ),
                    },
                )
            )
        if not bool(hot.any()):
            translations = torch.tensor([[0.0, 0.0, -8.0]], device=asset.device)
        else:
            pos = asset.data.body_pos_w[:, self._body_index]
            quat = asset.data.body_quat_w[:, self._body_index]
            local = self._offset.expand(pos.shape[0], -1, -1)
            world = pos + math_utils.quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape_as(pos)
            translations = world[hot]
        self._markers.visualize(translations=translations)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    color_over = os.environ.get("SFD_CRI_LINK_COLOR") == "1"
    if color_over and env_cfg.scene.num_envs == 1:
        env_cfg.viewer.eye = (1.8, -1.8, 1.3)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.6)
    # create isaac environment
    clip_spec = os.environ.get("SFD_PLAY_CLIPS", "").strip()
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video or clip_spec else None
    )
    if color_over or clip_spec:
        env = _CriLinkColorWrapper(env, threshold=0.96 if clip_spec else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    video_dir = os.path.join(log_dir, "videos", "play")
    if args_cli.video and not clip_spec:
        os.makedirs(video_dir, exist_ok=True)
        ckpt_stem = os.path.splitext(os.path.basename(resume_path))[0]
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"play_{ckpt_stem}_ovf_red" if color_over else f"play_{ckpt_stem}",
            "disable_logger": True,
        }
        print(f"[INFO] Recording play video to: {video_dir}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    last_cmd_key: torch.Tensor | None = None
    cri_log_path = os.environ.get("SFD_PLAY_MAX_CRI_CSV")
    cri_rows: list[list[float]] = []
    motion_path = os.environ.get("SFD_PLAY_MOTION_TXT")
    motion_rows: list[tuple[float, list[float], list[float], int]] = []
    motion_joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )

    def _motion_sample(base_env, time_s: float, section: int) -> None:
        robot = base_env.scene["robot"]
        names = list(robot.joint_names)
        index = [names.index(name) for name in motion_joint_names]
        q = robot.data.joint_pos[0, index].detach().cpu().tolist()
        qd = robot.data.joint_vel[0, index].detach().cpu().tolist()
        # The motion file has one extra fixed joint after the six arm joints.
        motion_rows.append((time_s, [float(v) for v in q] + [0.0], [float(v) for v in qd] + [0.0], section))

    if motion_path:
        _motion_sample(env.unwrapped, 0.0, 0)
    # Clips are "step:env" or "step:env:name". The camera follows that robot.
    # The default window is 3 s before the mark and 1.3 s after it.
    clip_marks: list[tuple[int, int, str]] = []
    if clip_spec:
        for part in clip_spec.split(","):
            bits = part.split(":")
            step = int(bits[0])
            env_index = int(bits[1])
            stem = bits[2] if len(bits) > 2 else f"ovf_step{step}_env{env_index}"
            clip_marks.append((step, env_index, stem))
    clip_before = int(os.environ.get("SFD_PLAY_CLIP_BEFORE", "45"))
    clip_after = int(os.environ.get("SFD_PLAY_CLIP_AFTER", "20"))
    clip_frames: list[list] = [[] for _ in clip_marks]
    clip_saved: list[bool] = [False for _ in clip_marks]
    clip_cam_env = -1
    clip_end = max((step + clip_after for step, _env, _stem in clip_marks), default=0)
    clip_dir = os.environ.get("SFD_PLAY_CLIP_DIR", "").strip() or video_dir
    _clip_font = None

    def _stamp_cri(shot, env_index: int):
        nonlocal _clip_font
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if _clip_font is None:
            _clip_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        cri = base.scene["robot"].data.CRI.detach()
        value = float(cri.reshape(cri.shape[0], -1)[env_index].amax())
        image = Image.fromarray(np.array(shot, copy=True))
        draw = ImageDraw.Draw(image)
        text = f"CRI {value:.3f}"
        bbox = draw.textbbox((0, 0), text, font=_clip_font)
        draw.text((image.width - (bbox[2] - bbox[0]) - 28, 22), text, font=_clip_font, fill=(0, 0, 0))
        return np.asarray(image)

    def _write_clip(index: int) -> None:
        import imageio.v2 as imageio

        frames = clip_frames[index]
        if clip_saved[index] or not frames:
            return
        _mark, env_index, stem = clip_marks[index]
        os.makedirs(clip_dir, exist_ok=True)
        path = os.path.join(clip_dir, f"{stem}.mp4")
        fps = max(1, int(round(1.0 / dt)))
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
        clip_frames[index] = []
        clip_saved[index] = True
        print(f"[INFO] OVF clip written: {path}  frames={len(frames)}", flush=True)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            base = env.unwrapped
            cm = getattr(base, "command_manager", None)
            if cm is not None and "ee_pose" in getattr(cm, "active_terms", []):
                cmd = cm.get_command("ee_pose")
                cmd_key = cmd[:, :3].detach()
                changed = last_cmd_key is None or not torch.allclose(cmd_key, last_cmd_key)
                if changed:
                    csv_utils.print_command_targets_from_env(
                        base, "ee_pose", label="play", max_rows=32
                    )
                    last_cmd_key = cmd_key.clone()
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if cri_log_path:
                cri = base.scene["robot"].data.CRI.detach()
                peak = cri.reshape(cri.shape[0], -1).amax(dim=-1).cpu().tolist()
                cri_rows.append([float(v) for v in peak])
            if motion_path:
                _motion_sample(base, timestep * dt + dt, 1)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video or cri_log_path or motion_path or clip_marks:
            timestep += 1
            if clip_marks:
                active = [
                    (index, env_index)
                    for index, (mark, env_index, _stem) in enumerate(clip_marks)
                    if mark - clip_before <= timestep <= mark + clip_after
                ]
                if active:
                    import numpy as np

                    cam = base.viewport_camera_controller
                    focus = active[0][1]
                    if focus != clip_cam_env:
                        cam.default_cam_eye = np.array([1.8, -1.8, 1.15])
                        cam.default_cam_lookat = np.array([0.0, 0.0, 0.45])
                        cam.cfg.env_index = focus
                        cam.update_view_to_asset_root("robot")
                        clip_cam_env = focus
                    else:
                        cam.update_view_to_asset_root("robot")
                    frame = base.render()
                    if frame is not None:
                        for index, env_index in active:
                            clip_frames[index].append(_stamp_cri(frame, env_index))
                    for index, (mark, _env_index, _stem) in enumerate(clip_marks):
                        if timestep >= mark + clip_after:
                            _write_clip(index)
            # Clips stop at the last window. A logged play stops at video_length
            # even when the viewport recorder is off.
            if clip_marks and timestep >= clip_end:
                break
            if not clip_marks and timestep >= args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if cri_log_path and cri_rows:
        os.makedirs(os.path.dirname(cri_log_path) or ".", exist_ok=True)
        with open(cri_log_path, "w", encoding="utf-8") as handle:
            n_env = len(cri_rows[0])
            handle.write("step,time_s," + ",".join(f"env{i}_max_cri" for i in range(n_env)) + "\n")
            for step, row in enumerate(cri_rows, start=1):
                handle.write(f"{step},{step * dt:.6f}," + ",".join(f"{v:.6f}" for v in row) + "\n")
        print(f"[INFO] Max CRI trace written: {cri_log_path}  steps={len(cri_rows)}")
    if motion_path and motion_rows:
        os.makedirs(os.path.dirname(motion_path) or ".", exist_ok=True)
        if len(motion_rows) > 1:
            last_t, last_q, _last_qd, _last_section = motion_rows[-1]
            motion_rows[-1] = (last_t, last_q, [0.0] * len(last_q), 0)
        with open(motion_path, "w", encoding="utf-8") as handle:
            for time_s, q, qd, section in motion_rows:
                cols = [f"{time_s:.10g}"] + [f"{v:.16g}" for v in q] + [f"{v:.16g}" for v in qd] + [str(section)]
                handle.write("\t".join(cols) + "\n")
        print(f"[INFO] Motion file written: {motion_path}  rows={len(motion_rows)}")
    if clip_marks:
        for index in range(len(clip_marks)):
            _write_clip(index)
        cam = getattr(env.unwrapped, "viewport_camera_controller", None)
        if cam is not None:
            cam.update_view_to_world()
    # close the simulator (flushes RecordVideo)
    env.close()
    if args_cli.video:
        print(f"[INFO] Play video written under: {video_dir}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
