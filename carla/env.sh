#!/usr/bin/env bash
set -euo pipefail

# Place as /workspace/DQNMedian50/DQNMedian50/carla/env.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/env"
CARLA_ROOT="${CARLA_ROOT:-/workspace/CARLA_0.9.14}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.7 || command -v python3 || command -v python)}"
CAPTURED_AT="$(date --iso-8601=seconds 2>/dev/null || date -Iseconds)"

mkdir -p "$ENV_DIR"

"$PYTHON_BIN" --version > "$ENV_DIR/python-version.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$ENV_DIR/pip-freeze.txt" 2>&1 || true
"$PYTHON_BIN" -m pip list > "$ENV_DIR/pip-list.txt" 2>&1 || true

cat > "$ENV_DIR/experiment-config.txt" <<'EOF'
CARLA DQN Experiment Configuration
==================================
CARLA Version: 0.9.14
Experiments: Epsilon Greedy and Median 50
Algorithm: DQN + RND + Count-Based Exploration
Components: Target Network, Replay Buffer, Adam Optimizer
Training Episodes: 500
Testing Episodes: 300
Maximum Episode Steps: 500
Learning Rate: 5e-5
Epsilon: 0.2
Gamma: 0.99
Batch Size: 64
Replay Capacity: 50000
Target Update Interval: 1000
RND Beta: 0.01
Count Beta: 0.05
Count State Bin Size: 1.0
Device: CUDA
Seed: 42
Convergence Threshold: 0.95
Analysis Block Size: 100 episodes
EOF

cat > "$ENV_DIR/run_command.txt" <<EOF
cd /workspace/DQNMedian50/DQNMedian50

export CARLA_ROOT="$CARLA_ROOT"
export PYTHONPATH="\$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg:\$CARLA_ROOT/PythonAPI/carla:\$PYTHONPATH"

mkdir -p /workspace/results/set4_v3

nohup "$PYTHON_BIN" -u -m carla.set4_v3july08 \\
    --train-episodes 500 \\
    --test-episodes 300 \\
    --max-episode-steps 500 \\
    --epsilon 0.2 \\
    --gamma 0.99 \\
    --raw-learning-rate 5e-5 \\
    --batch-size 64 \\
    --replay-capacity 50000 \\
    --target-update-interval 1000 \\
    --rnd-beta 0.01 \\
    --count-beta 0.05 \\
    --count-state-bin-size 1.0 \\
    --device cuda \\
    --seed 42 \\
    --output-dir /workspace/results/set4_v3 \\
    > /workspace/results/set4_v3/carla_set4_v3.log 2>&1 &
EOF

{
echo "CARLA EXPERIMENT ENVIRONMENT"
echo "============================"
echo "Captured: $CAPTURED_AT"
echo "Project root: $PROJECT_ROOT"
echo "CARLA root: $CARLA_ROOT"
echo "User: $(whoami 2>/dev/null || true)"
echo "Hostname: $(hostname 2>/dev/null || true)"
echo "Python: $PYTHON_BIN"
echo "Virtual environment: ${VIRTUAL_ENV:-not active}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not active}"

echo
echo "===== CARLA INSTALLATION ====="
if [ -d "$CARLA_ROOT" ]; then
    ls -ld "$CARLA_ROOT"
    find "$CARLA_ROOT" -maxdepth 3 \
        \( -name 'CarlaUE4.sh' -o -name 'CarlaUE4' -o -name 'CarlaUE4-Linux-Shipping' -o -name 'carla-*.egg' \) \
        -print 2>/dev/null
else
    echo "CARLA directory not found"
fi

echo
echo "===== CARLA PYTHON API ====="
CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"
PYTHONPATH="$CARLA_EGG:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version)
print("Executable:", sys.executable)
try:
    import carla
    print("CARLA API imported successfully")
    print("CARLA module:", carla.__file__)
except Exception as exc:
    print("CARLA import failed:", repr(exc))
for name in ("numpy", "pandas", "matplotlib", "torch"):
    try:
        module = __import__(name)
        print(name, "version:", getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "unavailable:", repr(exc))
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass
PY

echo
echo "===== CARLA PROCESS ====="
ps aux | grep -E 'CarlaUE4|CARLA' | grep -v grep || echo "CARLA server is not running"

echo
echo "===== NVIDIA GPU ====="
nvidia-smi 2>&1 || echo "nvidia-smi unavailable"

echo
echo "===== CUDA TOOLKIT ====="
nvcc --version 2>/dev/null || echo "nvcc unavailable"

echo
echo "===== OPERATING SYSTEM ====="
cat /etc/os-release 2>/dev/null || true
uname -a 2>/dev/null || true

echo
echo "===== CPU ====="
lscpu 2>/dev/null || true

echo
echo "===== MEMORY ====="
free -h 2>/dev/null || true

echo
echo "===== DISK ====="
df -h "$PROJECT_ROOT" 2>/dev/null || true

echo
echo "===== PROJECT FILES ====="
find "$PROJECT_ROOT" -maxdepth 2 -type f -printf '%P\n' 2>/dev/null | sort
} > "$ENV_DIR/system-info.txt" 2>&1

if command -v conda >/dev/null 2>&1; then
    conda list > "$ENV_DIR/conda-list.txt" 2>&1 || true
    conda env export --no-builds > "$ENV_DIR/environment.yml" 2>&1 || true
fi

if command -v git >/dev/null 2>&1 && git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$PROJECT_ROOT" rev-parse HEAD > "$ENV_DIR/git-commit.txt" 2>&1 || true
    git -C "$PROJECT_ROOT" status --short > "$ENV_DIR/git-status.txt" 2>&1 || true
fi

find "$PROJECT_ROOT" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null \
    | sort > "$ENV_DIR/source-file-list.txt" || true

(
    cd "$ENV_DIR"
    find . -type f ! -name SHA256SUMS.txt -print0 | sort -z \
        | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null || true
)

echo "CARLA environment details saved in: $ENV_DIR"
ls -lrth "$ENV_DIR"

