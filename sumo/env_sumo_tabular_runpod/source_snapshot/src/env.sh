#!/usr/bin/env bash
set -u

# Capture the current SUMO/Python environment, machine details, project revision,
# and the exact experiment launch command in a portable evidence folder.
#
# Usage:
#   ./capture_sumo_environment.sh
#
# With an exact run command:
#   ./capture_sumo_environment.sh env_runpod \
#     "python -u src/sumo_tabular_median50.py --train-episodes 500 ..."
#
# Or:
#   RUN_COMMAND='python -u src/sumo_tabular_median50.py ...' \
#     ./capture_sumo_environment.sh env_runpod

CAPTURE_DIR="${1:-env_runpod}"
CLI_RUN_COMMAND="${2:-}"
CAPTURED_AT="$(date --iso-8601=seconds 2>/dev/null || date -Iseconds)"
PROJECT_DIR="$(pwd)"
mkdir -p "$CAPTURE_DIR"

run_safely() {
    local outfile="$1"
    shift
    {
        echo "\$ $*"
        "$@"
    } >"$outfile" 2>&1 || true
}

{
    echo "SUMO EXPERIMENT ENVIRONMENT EVIDENCE"
    echo "===================================="
    echo "Captured: $CAPTURED_AT"
    echo "Project directory: $PROJECT_DIR"
    echo "Current user: $(whoami 2>/dev/null || echo unknown)"
    echo "UID/GID: $(id 2>/dev/null || echo unavailable)"
    echo "Hostname/container ID: $(hostname 2>/dev/null || echo unknown)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-not active}"
    echo "Virtual environment: ${VIRTUAL_ENV:-not active}"
    echo "Python executable: $(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo unavailable)"
    echo "SUMO_HOME: ${SUMO_HOME:-not set}"
    echo "PATH: ${PATH:-}"
    echo "PYTHONPATH: ${PYTHONPATH:-not set}"
} > "$CAPTURE_DIR/capture-summary.txt"

# Exact run command: explicit argument/environment variable takes priority.
RUN_COMMAND_VALUE="${CLI_RUN_COMMAND:-${RUN_COMMAND:-}}"
if [ -n "$RUN_COMMAND_VALUE" ]; then
    printf '%s\n' "$RUN_COMMAND_VALUE" > "$CAPTURE_DIR/run_command.txt"
else
    {
        echo "# No explicit run command was supplied."
        echo "# Relevant commands recovered from shell history:"
        history 2>/dev/null | grep -E 'python|nohup|sumo_tabular|sumo_dqn' | tail -n 50 || true
    } > "$CAPTURE_DIR/run_command.txt"
fi

# Preserve recent shell commands separately as supporting evidence.
{
    echo "# Relevant recent shell history captured at $CAPTURED_AT"
    history 2>/dev/null | grep -E 'python|nohup|sumo|rclone|source|activate|export SUMO_HOME|PYTHONPATH' | tail -n 200 || true
} > "$CAPTURE_DIR/relevant_shell_history.txt"

# Python/package environment.
PYTHON_BIN="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
if [ -n "$PYTHON_BIN" ]; then
    run_safely "$CAPTURE_DIR/python-version.txt" "$PYTHON_BIN" --version
    "$PYTHON_BIN" -m pip freeze > "$CAPTURE_DIR/pip-freeze.txt" 2>&1 || true
    "$PYTHON_BIN" -m pip list --format=columns > "$CAPTURE_DIR/pip-list.txt" 2>&1 || true

    "$PYTHON_BIN" - <<'PY' > "$CAPTURE_DIR/python-runtime-info.txt" 2>&1 || true
import os, platform, sys
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("Architecture:", platform.machine())
print("Prefix:", sys.prefix)
print("Base prefix:", getattr(sys, "base_prefix", "unknown"))
print("Virtual environment:", os.environ.get("VIRTUAL_ENV", "not active"))
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "not active"))

for module_name in ("numpy", "pandas", "matplotlib", "gymnasium", "traci", "sumolib", "torch"):
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "installed; version unavailable")
        print(f"{module_name}: {version}")
    except Exception as exc:
        print(f"{module_name}: unavailable ({exc})")

try:
    import torch
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Torch GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass
PY
fi

# Conda metadata, only when conda exists.
if command -v conda >/dev/null 2>&1; then
    conda list --explicit > "$CAPTURE_DIR/conda-explicit.txt" 2>&1 || true
    conda env export --no-builds > "$CAPTURE_DIR/environment.yml" 2>&1 || true
    conda env export --from-history > "$CAPTURE_DIR/environment-from-history.yml" 2>&1 || true
    conda list > "$CAPTURE_DIR/conda-list.txt" 2>&1 || true
    conda info --all > "$CAPTURE_DIR/conda-info.txt" 2>&1 || true
else
    echo "Conda is not installed or is not on PATH." > "$CAPTURE_DIR/conda-info.txt"
fi

# Re-creation helper for ordinary venvs.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    {
        echo "# Original virtual environment"
        echo "$VIRTUAL_ENV"
        echo
        echo "# Recreate approximately with:"
        echo "python3 -m venv sumoenv"
        echo "source sumoenv/bin/activate"
        echo "python -m pip install --upgrade pip"
        echo "python -m pip install -r pip-freeze.txt"
    } > "$CAPTURE_DIR/venv-recreation.txt"
fi

# SUMO binaries and Python bindings.
run_safely "$CAPTURE_DIR/sumo-version.txt" sumo --version
run_safely "$CAPTURE_DIR/netconvert-version.txt" netconvert --version
run_safely "$CAPTURE_DIR/sumo-binary-location.txt" bash -lc \
    'command -v sumo; command -v sumo-gui; command -v netconvert; command -v duarouter'
if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W 'sumo*' > "$CAPTURE_DIR/sumo-system-packages.txt" 2>&1 || true
fi

# OS and hardware evidence.
run_safely "$CAPTURE_DIR/os-release.txt" cat /etc/os-release
run_safely "$CAPTURE_DIR/kernel.txt" uname -a
run_safely "$CAPTURE_DIR/cpu.txt" lscpu
run_safely "$CAPTURE_DIR/memory.txt" free -h
run_safely "$CAPTURE_DIR/disk.txt" df -h
run_safely "$CAPTURE_DIR/network.txt" bash -lc \
    'hostname -I 2>/dev/null; ip addr 2>/dev/null; ip route 2>/dev/null'
run_safely "$CAPTURE_DIR/pci-devices.txt" lspci
run_safely "$CAPTURE_DIR/gpu-nvidia-smi.txt" nvidia-smi
run_safely "$CAPTURE_DIR/gpu-query.txt" nvidia-smi \
    --query-gpu=name,uuid,serial,driver_version,memory.total,pci.bus_id \
    --format=csv,noheader
run_safely "$CAPTURE_DIR/cuda-version.txt" bash -lc \
    'nvcc --version 2>/dev/null || true; cat /usr/local/cuda/version.json 2>/dev/null || true'

# Hardware identifiers may be hidden or provider-owned in a container.
{
    echo "These identifiers may be unavailable, virtualized, or shared by the cloud provider."
    echo
    for file in \
        /sys/class/dmi/id/product_name \
        /sys/class/dmi/id/product_uuid \
        /sys/class/dmi/id/product_serial \
        /sys/class/dmi/id/board_name \
        /sys/class/dmi/id/board_serial \
        /sys/class/dmi/id/bios_vendor \
        /sys/class/dmi/id/bios_version; do
        echo "[$file]"
        cat "$file" 2>/dev/null || echo "unavailable"
        echo
    done
} > "$CAPTURE_DIR/hardware-identifiers.txt"

# Git/project evidence.
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short > "$CAPTURE_DIR/git-status.txt" 2>&1 || true
    git rev-parse HEAD > "$CAPTURE_DIR/git-commit.txt" 2>&1 || true
    git remote -v > "$CAPTURE_DIR/git-remotes.txt" 2>&1 || true
    git log -1 --format=fuller > "$CAPTURE_DIR/git-last-commit.txt" 2>&1 || true
fi

# Record source inventory and copy important experiment scripts without copying results.
find src -maxdepth 2 -type f -printf '%P\n' 2>/dev/null | sort > "$CAPTURE_DIR/source-file-list.txt" || true
mkdir -p "$CAPTURE_DIR/source_snapshot"
find src -maxdepth 2 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.xml' \) \
    -exec cp --parents '{}' "$CAPTURE_DIR/source_snapshot/" \; 2>/dev/null || true

# Hashes provide tamper-evident fingerprints.
(
    cd "$CAPTURE_DIR" || exit 0
    find . -type f ! -name 'SHA256SUMS.txt' -print0 \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null || true
)

# Create a compressed archive alongside the folder.
ARCHIVE_NAME="${CAPTURE_DIR%/}.tar.gz"
tar -czf "$ARCHIVE_NAME" "$CAPTURE_DIR" 2>/dev/null || true

echo "Environment evidence saved to:"
echo "  $PROJECT_DIR/$CAPTURE_DIR"
echo "Archive saved to:"
echo "  $PROJECT_DIR/$ARCHIVE_NAME"
echo
echo "Important: upload both the evidence archive and experiment results to permanent storage."

