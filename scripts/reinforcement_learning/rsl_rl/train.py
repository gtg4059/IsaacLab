# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--reset_critic",
    action="store_true",
    default=False,
    help="After loading a checkpoint, reinitialize the critic MLP. The actor and the observation normalizer stay.",
)
parser.add_argument(
    "--warmup_iters",
    type=int,
    default=0,
    help="Number of initial iterations that update only the critic. The actor is frozen and the adaptive KL schedule is paused.",
)
parser.add_argument(
    "--warmup_value_loss",
    type=float,
    default=None,
    help="Keep updating only the critic until its mean value loss falls below this value. Std stays fixed if --freeze_std is set.",
)
parser.add_argument(
    "--freeze_std",
    action="store_true",
    default=False,
    help="Freeze the loaded per-joint action std. Does not replace it with the logged mean. Later updates change the action mean only.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _wrap_runner_save_with_curriculum_state(runner, env) -> None:
    """Persist ``common_step_counter`` in checkpoint ``infos`` for curriculum resume."""

    original_save = runner.save

    def save_with_curriculum_state(path: str, infos: dict | None = None) -> None:
        if infos is None:
            infos = {}
        isaac_env = env.unwrapped
        if hasattr(isaac_env, "common_step_counter"):
            infos["common_step_counter"] = int(isaac_env.common_step_counter)
        original_save(path, infos)

    runner.save = save_with_curriculum_state


def _resume_curriculum_from_checkpoint(
    env, runner, agent_cfg: RslRlBaseRunnerCfg, resume_infos: dict | None = None
) -> None:
    """Restore ``common_step_counter`` and re-apply curriculum terms after checkpoint load."""
    isaac_env = env.unwrapped
    if not hasattr(isaac_env, "curriculum_manager") or isaac_env.curriculum_manager is None:
        return

    if resume_infos is not None and "common_step_counter" in resume_infos:
        resumed_steps = int(resume_infos["common_step_counter"])
    else:
        resumed_steps = runner.current_learning_iteration * agent_cfg.num_steps_per_env

    isaac_env.common_step_counter = resumed_steps
    isaac_env.curriculum_manager.compute()
    print(f"[INFO] Resumed curriculum at common_step_counter={resumed_steps}")


def _set_optimizer_lr(runner, learning_rate: float, schedule: str | None = None) -> None:
    """Write the learning rate into PPO and every optimizer param group."""
    runner.alg.learning_rate = float(learning_rate)
    if schedule is not None:
        runner.alg.schedule = schedule
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate


def _install_lr_switch(runner, agent_cfg) -> None:
    """Apply the runner learning rate, or switch it once at ``lr_switch_iter``.

    Tasks without ``lr_switch_iter`` (or with it set to None) keep a single
    learning rate. Before the switch iteration, ``algorithm.learning_rate`` and
    ``algorithm.schedule`` apply and an adaptive schedule is free to move the
    rate. At that iteration and after, ``lr_switch_learning_rate`` and
    ``lr_switch_schedule`` are written on every update.
    """
    switch_iter = getattr(agent_cfg, "lr_switch_iter", None)
    if switch_iter is None:
        lr = float(agent_cfg.algorithm.learning_rate)
        _set_optimizer_lr(runner, lr)
        print(f"[INFO] Learning rate set to {lr:.3e}.")
        return

    switch_iter = int(switch_iter)
    early_lr = float(agent_cfg.algorithm.learning_rate)
    early_schedule = str(agent_cfg.algorithm.schedule)
    late_lr = float(agent_cfg.lr_switch_learning_rate)
    late_schedule = str(agent_cfg.lr_switch_schedule)
    start_it = int(runner.current_learning_iteration)
    state = {"n": 0, "announced_late": start_it >= switch_iter}

    if start_it < switch_iter:
        _set_optimizer_lr(runner, early_lr, early_schedule)
        print(
            f"[INFO] Learning rate {early_lr:.3e}, schedule {early_schedule} until iteration {switch_iter}."
        )
    else:
        _set_optimizer_lr(runner, late_lr, late_schedule)
        print(f"[INFO] Iteration {start_it}: learning rate {late_lr:.3e}, schedule {late_schedule}.")

    original_update = runner.alg.update

    def update_with_lr_switch():
        it = start_it + state["n"]
        state["n"] += 1
        if it < switch_iter:
            if runner.alg.schedule != early_schedule:
                runner.alg.schedule = early_schedule
        else:
            _set_optimizer_lr(runner, late_lr, late_schedule)
            if not state["announced_late"]:
                state["announced_late"] = True
                print(f"[INFO] Iteration {it}: learning rate {late_lr:.3e}, schedule {late_schedule}.")
        return original_update()

    runner.alg.update = update_with_lr_switch


def _reset_critic(runner) -> None:
    """Draw new critic MLP weights. Keep the actor and the critic's observation normalizer."""
    critic = runner.alg.critic
    mlp = getattr(critic, "mlp", None)
    if mlp is None:
        raise RuntimeError("Critic has no MLP to reinitialize.")
    n_linear = 0
    for module in mlp.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
            n_linear += 1
    critic_params = {id(p) for p in critic.parameters()}
    optimizer = runner.alg.optimizer
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in critic_params:
                optimizer.state.pop(param, None)
    print(f"[INFO] Reinitialized critic MLP ({n_linear} linear layers). Actor weights were kept.")


def _freeze_actor_std(runner) -> set[int]:
    """Stop the loaded per-joint action std from learning. Leave its values unchanged."""
    dist = getattr(runner.alg.actor, "distribution", None)
    if dist is None:
        raise RuntimeError("Actor has no action distribution to freeze.")
    if getattr(dist, "std_type", "scalar") == "log":
        param = dist.log_std_param
    else:
        param = dist.std_param
    param.requires_grad_(False)
    frozen_ids = {id(param)}
    for group in runner.alg.optimizer.param_groups:
        for group_param in group["params"]:
            if id(group_param) in frozen_ids:
                runner.alg.optimizer.state.pop(group_param, None)
    values = param.detach().float().view(-1).cpu()
    shown = ", ".join(f"{v:.4f}" for v in values.tolist())
    print(
        f"[INFO] Actor std frozen at the loaded per-joint values [{shown}] "
        f"(mean {float(values.mean()):.4f}). Updates will change the action mean only."
    )
    return frozen_ids


def _install_critic_warmup(
    runner, warmup_iters: int, value_limit: float | None, frozen_std_ids: set[int]
) -> None:
    """Update only the critic, then release the action mean. A frozen std stays frozen."""
    alg = runner.alg
    original_update = alg.update
    state = {"left": 0 if value_limit is not None else int(warmup_iters), "by_value": value_limit is not None}

    def _set_actor_mean_grad(enabled: bool) -> None:
        for param in alg.actor.parameters():
            if id(param) in frozen_std_ids:
                param.requires_grad_(False)
            else:
                param.requires_grad_(enabled)

    def update_critic_only():
        critic_only = state["by_value"] or state["left"] > 0
        saved_kl = alg.desired_kl
        if critic_only:
            alg.desired_kl = None
            _set_actor_mean_grad(False)
        try:
            loss_dict = original_update()
        finally:
            if critic_only:
                alg.desired_kl = saved_kl
            _set_actor_mean_grad(True)
        if critic_only:
            value = float(loss_dict["value"])
            if state["by_value"]:
                if value < float(value_limit):
                    state["by_value"] = False
                    print(
                        f"[INFO] Critic warmup finished. Value loss {value:.3f} is below {value_limit}. "
                        "Actor mean updates resumed."
                    )
                else:
                    print(f"[INFO] Critic warmup continues. Value loss {value:.3f}, target below {value_limit}.")
            else:
                state["left"] -= 1
                if state["left"] == 0:
                    print("[INFO] Critic warmup finished. Actor mean updates resumed.")
        return loss_dict

    alg.update = update_critic_only
    if value_limit is not None:
        print(
            f"[INFO] Critic-only warmup until value loss < {value_limit}. "
            "Actor mean and adaptive KL stay paused."
        )
    else:
        print(f"[INFO] Critic-only warmup for {warmup_iters} iterations. Actor and adaptive KL are paused.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    # The switch fields are read here. rsl_rl splats ``algorithm`` into PPO and rejects unknown keys.
    train_cfg = agent_cfg.to_dict()
    for key in ("lr_switch_iter", "lr_switch_learning_rate", "lr_switch_schedule"):
        train_cfg.pop(key, None)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # Episode rewards, metrics, curriculum, and terminations stay in TensorBoard.
    _log = runner.logger.log

    def _log_without_episode_extras(*args, **kwargs):
        kwargs["print_minimal"] = True
        return _log(*args, **kwargs)

    runner.logger.log = _log_without_episode_extras
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    _wrap_runner_save_with_curriculum_state(runner, env)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        resume_infos = runner.load(resume_path)
        _resume_curriculum_from_checkpoint(env, runner, agent_cfg, resume_infos)

    # Force the configured learning rate over the value stored in a checkpoint.
    _install_lr_switch(runner, agent_cfg)

    if args_cli.reset_critic:
        _reset_critic(runner)
    frozen_std_ids: set[int] = set()
    if args_cli.freeze_std:
        frozen_std_ids = _freeze_actor_std(runner)
    if args_cli.warmup_iters > 0 or args_cli.warmup_value_loss is not None:
        _install_critic_warmup(runner, args_cli.warmup_iters, args_cli.warmup_value_loss, frozen_std_ids)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
