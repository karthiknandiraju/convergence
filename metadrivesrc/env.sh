#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/DQNMedian50/convergence/metadrive"
ENV_DIR="$PROJECT_ROOT/env"
mkdir -p "$ENV_DIR"

# Search for the original experiment interpreter.
PYTHON_BIN=""

CANDIDATES=(
    "$PROJECT_ROOT/src/metadrive_py311_env/bin/python"
    "$PROJECT_ROOT/metadrive_py311_env/bin/python"
    "$PROJECT_ROOT/metadrive_env/bin/python"
    "/workspace/metadrive_py311_env/bin/python"
)

for CANDIDATE in "${CANDIDATES[@]}"; do
    if [[ -x "$CANDIDATE" ]] &&
       "$CANDIDATE" -c "import metadrive, torch, numpy" >/dev/null 2>&1; then
        PYTHON_BIN="$CANDIDATE"
        break
    fi
done

# ---------------------------------------------------------------------------
# Python and package information
# ---------------------------------------------------------------------------

if [[ -n "$PYTHON_BIN" ]]; then
    echo "Found original MetaDrive-compatible Python:"
    echo "$PYTHON_BIN"

    "$PYTHON_BIN" --version \
        > "$ENV_DIR/python-version.txt" 2>&1

    "$PYTHON_BIN" -m pip freeze \
        > "$ENV_DIR/pip-freeze.txt" 2>&1

    "$PYTHON_BIN" -m pip list \
        > "$ENV_DIR/pip-list.txt" 2>&1

    PACKAGE_SOURCE="Live capture from $PYTHON_BIN"
else
    echo "Original MetaDrive Python was not found."
    echo "Creating metadata from recorded experiment history."

    cat > "$ENV_DIR/python-version.txt" <<'EOF'
Python 3.11

Reconstruction note:
The completed experiment used the Python 3.11 virtual environment
metadrive_py311_env. The exact Python patch version was not retained.
EOF

    cat > "$ENV_DIR/pip-freeze.txt" <<'EOF'
# RECONSTRUCTED DIRECT DEPENDENCIES
# This is not the original complete pip freeze.

metadrive-simulator==0.4.3
torch==2.6.0
numpy<2
gymnasium
pandas
matplotlib
psutil

# PyTorch CUDA 12.4 wheel source:
# https://download.pytorch.org/whl/cu124
EOF

    cat > "$ENV_DIR/pip-list.txt" <<'EOF'
RECONSTRUCTED DIRECT PACKAGE INVENTORY
======================================

Package                 Version
----------------------- -------------------------
metadrive-simulator     0.4.3
torch                   2.6.0, CUDA 12.4 wheel
numpy                   below 2
gymnasium               version not retained
pandas                  version not retained
matplotlib              version not retained
psutil                  version not retained
EOF

    PACKAGE_SOURCE="Reconstructed from recorded experiment history"
fi

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

cat > "$ENV_DIR/experiment-config.txt" <<'EOF'
MetaDrive DQN Experiment Configuration
======================================

Environment: MetaDrive 0.4.3
Original virtual environment: metadrive_py311_env
Python family: 3.11
Device: CUDA

Exploration methods:
1. Epsilon Greedy
2. Median 50

Learning components:
- Deep Q-Network
- Target network
- Replay buffer
- Adam optimizer
- Random Network Distillation
- Count-based intrinsic reward
- Frozen-network testing

Training episodes: 500
Frozen-test episodes: 300
Maximum episode steps: 500

Epsilon: 0.2
Learning rate: 5e-5
Gamma: 0.99
Batch size: 64
Replay capacity: 50000
Target update interval: 1000
Hidden size: 128

RND beta: 0.01
RND learning rate: 1e-4
RND output size: 64

Count beta: 0.05
Count state bin size: 0.25

Discrete steering dimension: 3
Discrete throttle dimension: 3
Map blocks: 3
Traffic density: 0.20
Accident probability: 0.00

Success reward: 10
Collision penalty magnitude: 50
Out-of-road penalty magnitude: 10

Training seed: 42
Test seed base: 100000

Convergence threshold fraction: 0.95
Convergence window: 10
RMST event: collision
RMST horizon: 500
EOF

# ---------------------------------------------------------------------------
# Preserve the recorded run command
# ---------------------------------------------------------------------------

if [[ -f "$PROJECT_ROOT/runcommand.txt" ]]; then
    cp "$PROJECT_ROOT/runcommand.txt" \
       "$ENV_DIR/run_command.txt"
else
    cat > "$ENV_DIR/run_command.txt" <<'EOF'
cd /workspace/DQNMedian50/convergence/metadrive/src

source metadrive_py311_env/bin/activate

OUTPUT="/workspace/DQNMedian50/convergence/metadrive/results/results_metadrive_dqn_500_300_500"
mkdir -p "$OUTPUT"

nohup python -u metadrive_dqn_median50.py \
  --train-episodes 500 \
  --test-episodes 300 \
  --max-episode-steps 500 \
  --epsilon 0.2 \
  --learning-rate 5e-5 \
  --gamma 0.99 \
  --batch-size 64 \
  --replay-capacity 50000 \
  --target-update-steps 1000 \
  --hidden-size 128 \
  --rnd-beta 0.01 \
  --rnd-learning-rate 1e-4 \
  --rnd-output-size 64 \
  --count-beta 0.05 \
  --count-state-bin-size 0.25 \
  --discrete-steering-dim 3 \
  --discrete-throttle-dim 3 \
  --map-blocks 3 \
  --traffic-density 0.2 \
  --accident-prob 0.0 \
  --success-reward 10 \
  --collision-penalty 50 \
  --out-of-road-penalty 10 \
  --seed 42 \
  --test-seed 100000 \
  --device cuda \
  --convergence-threshold-fraction 0.95 \
  --convergence-window 10 \
  --rmst-event collision \
  --rmst-tau 500 \
  --output-dir "$OUTPUT" \
  > "$OUTPUT/metadrive_500_300_500.log" 2>&1 &

echo $! | tee "$OUTPUT/metadrive.pid"
tail -f "$OUTPUT/metadrive_500_300_500.log"
EOF
fi

# ---------------------------------------------------------------------------
# Current project structure
# ---------------------------------------------------------------------------

(
    cd "$PROJECT_ROOT"

    find . -maxdepth 3 \
        -path "./.git" -prune -o \
        -path "./env" -prune -o \
        -path "./metadrive_env" -prune -o \
        -path "./src/metadrive_py311_env" -prune -o \
        -type f -print |
        sed 's#^\./##' |
        sort
) > "$ENV_DIR/folder-structure.txt"

# ---------------------------------------------------------------------------
# System and hardware information
# ---------------------------------------------------------------------------

{
    echo "METADRIVE EXPERIMENT ENVIRONMENT"
    echo "================================"
    echo
    echo "Captured: $(date --iso-8601=seconds 2>/dev/null || date -Iseconds)"
    echo "Project root: $PROJECT_ROOT"
    echo "Environment folder: $ENV_DIR"
    echo "Package source: $PACKAGE_SOURCE"
    echo "Original training seed: 42"
    echo "Original test seed base: 100000"

    if [[ -n "$PYTHON_BIN" ]]; then
        echo "Python executable: $PYTHON_BIN"

        echo
        echo "===== RUNTIME PACKAGES ====="

        "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import sys

print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)

for package in (
    "metadrive-simulator",
    "torch",
    "numpy",
    "gymnasium",
    "pandas",
    "matplotlib",
    "psutil",
):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except Exception:
        print(f"{package}: unavailable")

try:
    import torch

    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch check failed:", repr(exc))
PY
    else
        echo "Python executable: original interpreter not found"
        echo "Recorded Python family: 3.11"
        echo "Recorded MetaDrive version: 0.4.3"
        echo "Recorded PyTorch setup: 2.6.0 with CUDA 12.4 wheels"
    fi

    echo
    echo "===== NVIDIA GPU ====="
    nvidia-smi 2>&1 || echo "nvidia-smi unavailable"

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

} > "$ENV_DIR/system-info.txt" 2>&1

echo
echo "MetaDrive environment metadata created:"
echo "$ENV_DIR"
echo

find "$ENV_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%f\n' |
    sort
