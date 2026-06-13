#!/usr/bin/env bash
# =============================================================================
#  run_eda.sh
#  RFS-SCP v16.1 — EDA Runner
#
#  Runs the AnnoMI exploratory data analysis script.
#
#  Usage:
#    bash scripts/run_eda.sh [--annomi-dir <path>]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANNOMI_DIR="${ANNOMI_DIR:-${ROOT_DIR}/data/annomi}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annomi-dir) ANNOMI_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  RFS-SCP v16.1 — EDA"
echo "  AnnoMI dir : ${ANNOMI_DIR}"
echo "============================================================"

python "${ROOT_DIR}/src/eda/annomi_eda.py" \
  --annomi-dir "${ANNOMI_DIR}"
