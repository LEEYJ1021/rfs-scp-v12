#!/usr/bin/env bash
# =============================================================================
#  run_validation.sh
#  RFS-SCP v16.1 — Concern Validation Script Runner
#
#  Runs src/validate_concerns.py to independently verify that all v16.1
#  concern responses are correctly implemented in the codebase.
#
#  Checks performed:
#    1. VADER backend is real (not a silent fallback)
#    2. AUC floor = 0.55 with Tanana et al. (2016) citation
#    3. MCMC has no temperature constant (temp=300 removed)
#    4. Hold-out weight generalisation gaps reported in results CSV
#    5. Communication proxy coverage < 10% (justifies FUTURE WORK demotion)
#    6. src/rfs_scp_v16_main.py is present and committed
#    7. JSAI domestic conference citation uses year 2026
#
#  Usage:
#    bash scripts/run_validation.sh [--annomi-dir <path>] [--results-dir <path>]
#
#  Returns exit code 0 if all checks pass, 1 otherwise.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANNOMI_DIR="${ANNOMI_DIR:-${ROOT_DIR}/data/annomi}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annomi-dir)   ANNOMI_DIR="$2";   shift 2 ;;
    --results-dir)  RESULTS_DIR="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  RFS-SCP v16.1 — Concern Validation"
echo "  Results dir : ${RESULTS_DIR}"
echo "============================================================"

python "${ROOT_DIR}/src/validate_concerns.py" \
  --annomi-dir "${ANNOMI_DIR}" \
  --results-dir "${RESULTS_DIR}"
