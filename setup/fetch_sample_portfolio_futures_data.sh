#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${PROJECT_ROOT}/venv" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/venv/bin/activate"
fi

# Always run from project root
cd "${PROJECT_ROOT}"

echo "========== Last 10 years =========="
python yahoo_finance/get_data.py --years 10

echo "========== Last 20 years =========="
python yahoo_finance/get_data.py --years 20

echo "========== 2000 to 2015 =========="
python yahoo_finance/get_data.py --from 2000-01-01 --to 2015-12-31

echo "✅ All downloads complete."

