#!/usr/bin/env bash
set -euo pipefail

# Place as /workspace/DQNMedian50/DQNMedian50/sumo/env.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/env"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
CAPTURED_AT="$(date --iso-8601=seconds 2>/dev/null || date -Iseconds)"

mkdir -p "$ENV_DIR"

"$PYTHON_BIN" --version > "$ENV_DIR/python-version.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$ENV_DIR/pip-freeze.txt" 2>&1 || true
"$PYTHON_BIN" -m pip list > "$ENV_DIR/pip-list.txt" 2>&1 || true

cat > "$ENV_DIR/experiment-config.txt" <<'EOF'
SUMO Reinforcement-Learning Experiment Configuration
====================================================
Experiments: Epsilon Greedy and Median 50
Training Episodes: 500
Testing Episodes: 300
Maximum Episode Steps: 500
Epsilon: 0.2
Gamma: 0.99
Seed: 42
Analysis Block Size: 100 episodes
Convergence Threshold: 0.95
EOF

cat > "$ENV_DIR/run_command.txt" <<'EOF'
# Insert the exact command used for the relevant SUMO run below.
# Example tabular run:
cd /workspace/DQNMedian50/DQNMedian50/sumo
python3 src/sumo_tabular_median50.py \
    --train-episodes 500 \
    --test-episodes 300 \
    --max-episode-steps 500 \
    --epsilon 0.2 \
    --gamma 0.99 \
    --seed 42
EOF

{
echo "SUMO EXPERIMENT ENVIRONMENT"
echo "==========================="
echo "Captured: $CAPTURED_AT"
echo "Project root: $PROJECT_ROOT"
echo "User: $(whoami 2>/dev/null || true)"
echo "Hostname: $(hostname 2>/dev/null || true)"
echo "Python: $PYTHON_BIN"
echo "Virtual environment: ${VIRTUAL_ENV:-not active}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not active}"
echo "SUMO_HOME: ${SUMO_HOME:-not set}"

echo
echo "===== SUMO ====="
sumo --version 2>&1 || echo "SUMO executable unavailable"
echo "sumo: $(command -v sumo 2>/dev/null || echo unavailable)"
echo "sumo-gui: $(command -v sumo-gui 2>/dev/null || echo unavailable)"
echo "netconvert: $(command -v netconvert 2>/dev/null || echo unavailable)"

echo
echo "===== SUMO PYTHON MODULES ====="
"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version)
print("Executable:", sys.executable)
for name in ("traci", "sumolib", "numpy", "pandas", "matplotlib", "torch"):
    try:
        module = __import__(name)
        print(name, "version:", getattr(module, "__version__", "installed/version unavailable"))
        print(name, "path:", getattr(module, "__file__", "unknown"))
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

find "$PROJECT_ROOT/src" -maxdepth 2 -type f -printf '%P\n' 2>/dev/null \
    | sort > "$ENV_DIR/source-file-list.txt" || true

(
    cd "$ENV_DIR"
    find . -type f ! -name SHA256SUMS.txt -print0 | sort -z \
        | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null || true
)

tar -czf "$PROJECT_ROOT/env.tar.gz" -C "$PROJECT_ROOT" env

echo "SUMO environment details saved in: $ENV_DIR"
ls -lrth "$ENV_DIR"
echo "Archive created: $PROJECT_ROOT/env.tar.gz"
ls -lh "$PROJECT_ROOT/env.tar.gz"

