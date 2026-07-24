#!/usr/bin/env bash
# Run the modular policy with all user-supplied arguments.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${script_dir}/.." && pwd)"

python -u "${source_root}/policies/safetypool_modular.py" "$@"

