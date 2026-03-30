# KHU G1 Train

Isaac Lab 기반 Unitree G1 보행 학습 프로젝트입니다.

## 1) 프로젝트 개요

- 학습 스크립트: `scripts/rsl_rl/train.py`
- 재생(검증) 스크립트: `scripts/rsl_rl/play.py`
- 환경 등록: `source/g1_loco/g1_loco/tasks/manager_based/g1_loco/__init__.py`
- 주요 환경 설정 파일:
  - `rough_env_cfg.py` (Rough 기본)
  - `flat_env_cfg.py` (Flat 기본)
  - `g1_loco_env_cfg.py` (Rough Base)

## 2) 사전 준비

- 기준 문서: Isaac Lab v2.3.2 `Local Installation > System Requirements`
- OS: Ubuntu 22.04 (Linux x64) 또는 Windows 11 (x64)
- RAM: 32 GB 이상
- GPU VRAM: 16 GB 이상
- NVIDIA Driver: 최신 Production Branch 권장
  - Linux: `580.65.06` 이상 권장 (특히 Ubuntu 22.04.5 + kernel 6.8.0-48 이상)
  - Windows: `580.88` 권장
- Python (Isaac Sim 버전에 맞춰 선택)
  - Isaac Sim 5.x: Python 3.11
  - Isaac Sim 4.x: Python 3.10
- Git + Git LFS
- Conda(권장: Miniconda)

## 3) Miniconda 설치 (없을 때)

### 설치 확인

```bash
conda --version
```

### Linux/WSL 기준 설치 예시

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## 4) `env_isaaclab` 환경 생성

### 새 환경 생성 예시

```bash
# Isaac Sim 5.x 사용 시
conda create -n env_isaaclab python=3.11 -y

# IsaacLab sh 파일 사용시
./isaaclab.sh --conda my_env_name

# Activate environment
conda activate env_isaaclab
```

## 5) Isaac Sim 및 Lab 설치

Isaac Lab을 먼저 설치해야 이 프로젝트 스크립트가 동작합니다.

- 공식 git 저장소: https://github.com/isaac-sim/IsaacLab
- 공식 문서(v2.3.2): https://isaac-sim.github.io/IsaacLab/v2.3.2/index.html
- 설치 가이드: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html

## 6) Isaac Sim 환경 스크립트 적용 (`setup_conda_env.sh`)

Isaac Sim/Lab 설치 후, `env_isaaclab` 활성화만으로는 Isaac Sim 관련 환경변수가 자동 반영되지 않을 수 있습니다.
아래 두 방법 중 하나를 사용하세요.

### 방법 A: 매번 수동으로 실행

```bash
conda activate env_isaaclab
source /home/safetics/isaacsim/setup_conda_env.sh
```

### 방법 B(옵션): `conda activate` 시 자동 실행

한 번만 설정하면, 이후에는 `conda activate env_isaaclab`만으로 자동 적용됩니다.

```bash
conda activate env_isaaclab
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/isaacsim.sh" <<'EOF'
#!/usr/bin/env bash
source /home/safetics/isaacsim/setup_conda_env.sh
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/isaacsim.sh"
```

자동 적용 해제:

```bash
conda activate env_isaaclab
rm -f "$CONDA_PREFIX/etc/conda/activate.d/isaacsim.sh"
```

## 7) 저장소 클론

```bash
git clone https://github.com/safetics-dev/KHU_g1_train.git
cd KHU_g1_train
```

## 8) Git LFS 파일 받기

이 저장소의 G1 USD 자산은 Git LFS로 관리됩니다.

```bash
# git-lfs install
sudo apt install git-lfs

# 1회 설정
git lfs install

# 현재 저장소 LFS 파일 다운로드
git lfs pull
```

## 9) `g1_loco` 패키지 설치 (중요)

반드시 `env_isaaclab` 활성화 후 설치하세요.

```bash
conda activate env_isaaclab
cd /path/to/KHU_g1_train
python -m pip install -e source/g1_loco
```

### 왜 설치가 필요한가?

- `import g1_loco.tasks`가 실행되면서 `gym.register(...)`가 호출됩니다.
- 이 단계가 되어야 `gym.make("KHU-...")`에서 태스크 ID를 찾을 수 있습니다.
- `-e`(editable) 설치라서 코드 수정이 즉시 반영됩니다.
- 보통 **환경당 1회 설치**면 충분하고, 새 conda 환경을 만들면 다시 설치해야 합니다.

## 10) 학습/재생 실행

### Rough 기본 환경

```bash
# 학습
python scripts/rsl_rl/train.py --task KHU-Velocity-Rough-G1-v0 --num_envs 4096 --headless

# 재생
python scripts/rsl_rl/play.py --task KHU-Velocity-Rough-G1-Play-v0 --checkpoint path/to/model.pt
```

### Flat 기본 환경

```bash
# 학습
python scripts/rsl_rl/train.py --task KHU-Velocity-Flat-G1-v0 --num_envs 4096 --headless

# 재생
python scripts/rsl_rl/play.py --task KHU-Velocity-Flat-G1-Play-v0 --checkpoint path/to/model.pt
```

### Rough Base 환경 (`g1_loco_env_cfg` 기반)

```bash
# 학습
python scripts/rsl_rl/train.py --task KHU-Velocity-Base-G1-v0 --num_envs 4096 --headless

# 재생
python scripts/rsl_rl/play.py --task KHU-Velocity-Base-G1-Play-v0 --checkpoint path/to/model.pt
```

## 11) Gym 등록 매핑

| Task ID | Env Config | Runner Config |
|---|---|---|
| `KHU-Velocity-Rough-G1-v0` | `rough_env_cfg:G1RoughEnvCfg` | `rsl_rl_ppo_cfg:G1RoughPPORunnerCfg` |
| `KHU-Velocity-Rough-G1-Play-v0` | `rough_env_cfg:G1RoughEnvCfg_PLAY` | `rsl_rl_ppo_cfg:G1RoughPPORunnerCfg` |
| `KHU-Velocity-Flat-G1-v0` | `flat_env_cfg:G1FlatEnvCfg` | `rsl_rl_ppo_cfg:G1FlatPPORunnerCfg` |
| `KHU-Velocity-Flat-G1-Play-v0` | `flat_env_cfg:G1FlatEnvCfg_PLAY` | `rsl_rl_ppo_cfg:G1FlatPPORunnerCfg` |
| `KHU-Velocity-Base-G1-v0` | `g1_loco_env_cfg:G1LocoEnvCfg` | `rsl_rl_ppo_cfg:G1RoughPPORunnerCfg` |
| `KHU-Velocity-Base-G1-Play-v0` | `g1_loco_env_cfg:G1LocoEnvCfg_PLAY` | `rsl_rl_ppo_cfg:G1RoughPPORunnerCfg` |

## 12) 자주 발생하는 문제

### `ModuleNotFoundError: No module named 'g1_loco'`

```bash
conda activate env_isaaclab
python -m pip install -e source/g1_loco
```

### `ModuleNotFoundError: No module named 'isaacsim'`

`ModuleNotFoundError: No module named 'isaacsim'` 오류가 발생하면, 가상환경이 활성화되어 있는지 확인하고 `source /home/safetics/isaacsim/setup_conda_env.sh`를 실행하세요.

```bash
conda activate env_isaaclab
source /home/safetics/isaacsim/setup_conda_env.sh
```

### 에셋이 포인터 파일처럼 보이는 경우(LFS 미반영)

```bash
git lfs pull
```

### `conda: command not found`

Miniconda 설치 후 새 셸을 열거나 `source ~/.bashrc`를 실행하세요.
