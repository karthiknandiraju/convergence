#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "${script_dir}/../.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/safetypool_matplotlib_cache}"
mkdir -p "${MPLCONFIGDIR}"

python -u \
    "${package_root}/02_source_code/analysis/generate_ieee_results.py" \
    --package-root "${package_root}"
