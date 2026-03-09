# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--log_csv",
    type=str,
    default=None,
    help="If set, save robot joint data and command application timing to this CSV path (e.g. play_log.csv).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    # RslRlObsPaddingWrapper,
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # # If checkpoint has different obs dim (e.g. distillation 278 vs env 269), pad obs so load succeeds
    # try:
    #     ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
    # except TypeError:
    #     ckpt = torch.load(resume_path, map_location="cpu")
    # state_dict = ckpt.get("model_state_dict", ckpt)
    # if isinstance(state_dict, dict):
    #     if "student.0.weight" in state_dict:
    #         ckpt_obs_dim = state_dict["student.0.weight"].shape[1]
    #     elif "actor.0.weight" in state_dict:
    #         ckpt_obs_dim = state_dict["actor.0.weight"].shape[1]
    #     else:
    #         ckpt_obs_dim = None
    #     if ckpt_obs_dim is not None and ckpt_obs_dim > env.num_obs:
    #         env = RslRlObsPaddingWrapper(env, ckpt_obs_dim)
    #         print(f"[INFO]: Obs padding for checkpoint: env num_obs {env.env.num_obs} -> {ckpt_obs_dim}")
    # del ckpt

    # # Fallback: config num_obs_teacher (e.g. when checkpoint format differs)
    # if (
    #     getattr(agent_cfg.policy, "num_obs_teacher", None) is not None
    #     and agent_cfg.policy.num_obs_teacher > env.num_obs
    # ):
    #     env = RslRlObsPaddingWrapper(env, agent_cfg.policy.num_obs_teacher)
    #     print(
    #         f"[INFO]: Obs padding from config: env num_obs {env.env.num_obs} -> {agent_cfg.policy.num_obs_teacher}"
    #     )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    base_env = env.unwrapped

    # Optional CSV logging: env별로 별도 파일 (env_id 0 → name_env0.csv, env_id 1 → name_env1.csv, ...)
    csv_files = []
    csv_writers = []
    if args_cli.log_csv:
        log_path = os.path.abspath(args_cli.log_csv)
        log_dir = os.path.dirname(log_path) or "."
        log_base = os.path.splitext(os.path.basename(log_path))[0]
        os.makedirs(log_dir, exist_ok=True)
        robot = base_env.scene["robot"]
        num_envs = base_env.num_envs
        jpos = robot.data.joint_pos
        num_joints = jpos.shape[1]
        jnames = getattr(robot.data, "joint_names", None)
        j_prefix = (lambda i: jnames[i] if jnames and i < len(jnames) else f"j{i}")
        header = (
            ["step", "sim_time"]
            + [f"joint_pos_{j_prefix(i)}" for i in range(num_joints)]
            + [f"joint_vel_{j_prefix(i)}" for i in range(num_joints)]
            + ["root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z"]
            + ["root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"]
            + ["vel_cmd_x", "vel_cmd_y", "ang_vel_cmd_z"]
            + ["force_cmd_x", "force_cmd_y", "force_cmd_z", "force_active"]
        )
        for e in range(num_envs):
            fpath = os.path.join(log_dir, f"{log_base}_env{e}.csv")
            f = open(fpath, "w", newline="")
            w = csv.writer(f)
            w.writerow(header)
            csv_files.append(f)
            csv_writers.append(w)
        print(f"[INFO] Logging play data to {num_envs} CSV files: {log_dir}/{log_base}_env0.csv ... {log_base}_env{num_envs-1}.csv")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    log_step = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # CSV log: 각 env별 파일에 해당 로봇 데이터만 한 줄씩 기록
        if csv_writers:
            robot = base_env.scene["robot"]
            cm = base_env.command_manager
            sim_time = log_step * dt
            vel_cmd = cm.get_command("base_velocity").detach().cpu().numpy()
            force_cmd = cm.get_command("base_force").detach().cpu().numpy()
            force_term = cm.get_term("base_force")
            force_active = force_term.is_force_active.detach().cpu().numpy()
            jpos_np = robot.data.joint_pos.detach().cpu().numpy()
            jvel_np = robot.data.joint_vel.detach().cpu().numpy()
            root_lin = robot.data.root_lin_vel_b.detach().cpu().numpy()
            root_ang = robot.data.root_ang_vel_b.detach().cpu().numpy()
            for e in range(base_env.num_envs):
                row = (
                    [log_step, sim_time]
                    + jpos_np[e].tolist()
                    + jvel_np[e].tolist()
                    + root_lin[e].tolist()
                    + root_ang[e].tolist()
                    + vel_cmd[e].tolist()
                    + force_cmd[e].tolist()
                    + [int(force_active[e])]
                )
                csv_writers[e].writerow(row)
            for f in csv_files:
                f.flush()
            log_step += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if csv_files:
        for f in csv_files:
            f.close()
        log_dir = os.path.dirname(os.path.abspath(args_cli.log_csv)) or "."
        log_base = os.path.splitext(os.path.basename(args_cli.log_csv))[0]
        print(f"[INFO] CSV saved (steps={log_step}): {log_dir}/{log_base}_env0.csv ... env{len(csv_files)-1}.csv")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
