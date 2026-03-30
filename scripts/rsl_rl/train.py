# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL로 강화학습 에이전트를 학습하는 스크립트."""

"""Isaac Sim 앱을 실행한 뒤 학습을 진행한다."""

import argparse
import sys

from isaaclab.app import AppLauncher

# 로컬 모듈
import cli_args  # isort: skip

# CLI 인자 정의
parser = argparse.ArgumentParser(description="RSL-RL로 강화학습 에이전트를 학습합니다.")
parser.add_argument("--video", action="store_true", default=False, help="학습 중 영상을 기록합니다.")
parser.add_argument("--video_length", type=int, default=200, help="기록할 영상 길이(스텝).")
parser.add_argument("--video_interval", type=int, default=2000, help="영상 기록 주기(스텝).")
parser.add_argument("--num_envs", type=int, default=None, help="동시에 시뮬레이션할 환경 수.")
parser.add_argument("--task", type=str, default=None, help="실행할 태스크 이름.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL 에이전트 설정 엔트리포인트 이름.")
parser.add_argument("--seed", type=int, default=None, help="환경 시드 값.")
parser.add_argument("--max_iterations", type=int, default=None, help="정책 학습 반복 횟수.")
parser.add_argument("--distributed", action="store_true", default=False, help="멀티 GPU/노드 분산 학습 실행.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="IO descriptor를 내보냅니다.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help="Ray 연동 시 자동 설정되는 프로세스 ID.")

# RSL-RL 공통 인자 추가
cli_args.add_rsl_rl_args(parser)

# AppLauncher 인자 추가
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 영상 기록 시 카메라를 항상 활성화
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra에 전달할 인자만 남기도록 sys.argv 정리
sys.argv = [sys.argv[0]] + hydra_args

# Omniverse 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""지원하는 최소 RSL-RL 버전을 확인한다."""

import importlib.metadata as metadata
import platform

from packaging import version

# 최소 지원 rsl-rl 버전 확인
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

"""이후부터 학습 실행 본문."""

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

# 로거 설정
logger = logging.getLogger(__name__)

import g1_loco.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """RSL-RL 에이전트 학습 실행."""
    # Hydra 기본 설정을 일반 CLI 인자로 덮어쓴다.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations)

    # 구버전 설정 호환 처리
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 환경 시드 설정
    # 환경 초기화 과정에서도 랜덤화가 일어나므로 이 시점에 시드를 고정한다.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # CPU + 분산학습 조합은 지원하지 않으므로 사전 차단
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # 멀티 GPU 분산학습 설정
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 프로세스마다 다른 시드를 사용해 데이터 다양성 확보
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 실험 로그 디렉터리 경로 생성
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 실행별 로그 디렉터리: {timestamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 아래 출력 포맷은 Ray Tune 워크플로우에서 파싱하므로 변경하면 안 된다.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 요청된 경우 IO descriptor 내보내기 옵션 적용
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported.")

    # 환경 설정에 로그 디렉터리 반영(환경 타입 공통)
    env_cfg.log_dir = log_dir

    # Isaac 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # RL 알고리즘 요구사항에 맞게 필요 시 단일 에이전트 환경으로 변환
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 이어학습/증류학습이면 체크포인트 경로 확보
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 영상 기록 래퍼 적용
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

    # rsl-rl 인터페이스에 맞게 환경 래핑
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # rsl-rl 러너 생성
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    # 로그에 현재 git 상태 기록
    runner.add_git_repo_to_log(__file__)
    
    # 체크포인트 로드
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        # 기존 학습 가중치 복원
        runner.load(resume_path)

    # 사용한 환경/에이전트 설정을 로그에 저장
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 학습 실행
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # 시뮬레이터 종료
    env.close()


if __name__ == "__main__":
    # 메인 실행
    main()
    # 앱 종료
    simulation_app.close()
