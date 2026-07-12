#!/usr/bin/env bash
set -u

# Place this file in:
# /workspace/DQNMedian50/DQNMedian50/highway/env.sh
# Run:
# bash /workspace/DQNMedian50/DQNMedian50/highway/env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/env"
RESULTS_DIR="$PROJECT_ROOT/results_highway_dqn_500_300_500"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$ENV_DIR" "$RESULTS_DIR"
rm -f "$ENV_DIR/conda-explicit.txt" \
      "$ENV_DIR/conda-list.txt" \
      "$ENV_DIR/environment.yml" \
      "$ENV_DIR/environment-from-history.yml"

"$PYTHON_BIN" --version > "$ENV_DIR/python-version.txt" 2>&1
"$PYTHON_BIN" -m pip freeze > "$ENV_DIR/pip-freeze.txt" 2>&1
"$PYTHON_BIN" -m pip list > "$ENV_DIR/pip-list.txt" 2>&1

cat > "$ENV_DIR/experiment-config.txt" <<'EOF'
Highway DQN Experiment Configuration
====================================

Environment: highway-env / highway-v0

Experiments:
1. Epsilon Greedy
2. Median 50

Algorithm:
Deep Q-Network (DQN)
Target Network
Replay Buffer
Adam Optimizer

Training Episodes: 500
Testing Episodes: 300
Maximum Episode Steps: 500

Epsilon: 0.2
Gamma: 0.99
Batch Size: 64
Replay Capacity: 50000
Target Update Interval: 1000

Device: CUDA when available
Seed: 42
Convergence Threshold: 0.95
Analysis Block Size: 100 episodes
EOF

{
echo "Captured: $(date --iso-8601=seconds)"
echo

echo "===== HIGHWAY-ENV ====="
"$PYTHON_BIN" - <<'PY'
try:
    import highway_env
    print("highway-env version:", getattr(highway_env, "__version__", "unknown"))
    print("highway-env module:", highway_env.__file__)
except Exception as exc:
    print("highway-env import failed:", repr(exc))
PY

echo
echo "===== GYMNASIUM / GYM ====="
"$PYTHON_BIN" - <<'PY'
for name in ("gymnasium", "gym"):
    try:
        module = __import__(name)
        print(name, "version:", getattr(module, "__version__", "unknown"))
        print(name, "module:", module.__file__)
    except Exception as exc:
        print(name, "import failed:", repr(exc))
PY

echo
echo "===== ENVIRONMENT CREATION CHECK ====="
"$PYTHON_BIN" - <<'PY'
try:
    import gymnasium as gym
except ImportError:
    import gym
try:
    import highway_env  # registers highway environments
    env = gym.make("highway-v0")
    result = env.reset(seed=42)
    observation = result[0] if isinstance(result, tuple) else result
    print("highway-v0 created successfully")
    print("Observation shape:", getattr(observation, "shape", "unknown"))
    print("Action space:", env.action_space)
    env.close()
except Exception as exc:
    print("highway-v0 creation failed:", repr(exc))
PY

echo
echo "===== PYTHON ====="
"$PYTHON_BIN" --version
command -v "$PYTHON_BIN"
echo "Virtual environment: ${VIRTUAL_ENV:-unknown}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"

echo
echo "===== PIP ====="
"$PYTHON_BIN" -m pip --version

echo
echo "===== PYTORCH ====="
"$PYTHON_BIN" - <<'PY'
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
except Exception as exc:
    print("PyTorch check failed:", repr(exc))
PY

echo
echo "===== NUMPY / PANDAS / MATPLOTLIB ====="
"$PYTHON_BIN" - <<'PY'
for name in ("numpy", "pandas", "matplotlib"):
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "check failed:", repr(exc))
PY

echo
echo "===== NVIDIA GPU ====="
nvidia-smi 2>/dev/null || echo "nvidia-smi is unavailable"

echo
echo "===== CUDA TOOLKIT ====="
nvcc --version 2>/dev/null || echo "nvcc is not installed or not in PATH"

echo
echo "===== OPERATING SYSTEM ====="
cat /etc/os-release 2>/dev/null
uname -a

echo
echo "===== CPU ====="
lscpu 2>/dev/null

echo
echo "===== MEMORY ====="
free -h 2>/dev/null

echo
echo "===== DISK ====="
df -h "$PROJECT_ROOT"

echo
echo "===== PROJECT STRUCTURE ====="
find "$PROJECT_ROOT" -maxdepth 2 -type f -print 2>/dev/null | sort

} > "$ENV_DIR/system-info.txt" 2>&1

cat > "$ENV_DIR/run_command.txt" <<EOF
cd "$PROJECT_ROOT"

mkdir -p "$RESULTS_DIR"

nohup "$PYTHON_BIN" highway.py \\
    --train-episodes 500 \\
    --test-episodes 300 \\
    --max-episode-steps 500 \\
    --epsilon 0.2 \\
    --gamma 0.99 \\
    --batch-size 64 \\
    --device cuda \\
    --seed 42 \\
    --output-dir "$RESULTS_DIR" \\
    > "$RESULTS_DIR/highway_dqn_500_300_500.log" 2>&1 &

echo \$! > "$RESULTS_DIR/highway.pid"
EOF

cat > "$ENV_DIR/folder-structure.txt" <<'EOF'
highway/
├── env.sh
├── highway.py
├── env/
│   ├── experiment-config.txt
│   ├── folder-structure.txt
│   ├── pip-freeze.txt
│   ├── pip-list.txt
│   ├── python-version.txt
│   ├── run_command.txt
│   └── system-info.txt
└── results_highway_dqn_500_300_500/
    ├── all_episode_results.csv
    ├── all_experiments_learning_rate_summary.csv
    ├── all_experiments_runtime_logs.csv
    ├── all_experiments_test_episode_rewards.csv
    ├── all_experiments_train_episode_rewards.csv
    ├── config.json
    ├── figures_ieee/
    └── models/
EOF

echo "Highway environment details saved in: $ENV_DIR"
ls -lh "$ENV_DIR"

