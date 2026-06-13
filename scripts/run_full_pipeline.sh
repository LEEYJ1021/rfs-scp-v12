#!/usr/bin/env bash
# =============================================================================
#  run_full_pipeline.sh
#  RFS-SCP v16.1 — Full Pipeline Runner
#
#  Usage:
#    bash scripts/run_full_pipeline.sh [--annomi-dir <path>]
#
#  Environment variables:
#    ANNOMI_DIR   Override AnnoMI data directory (default: data/annomi)
#    RFS_OUT_v16  Override output directory    (default: results/)
#
#  Notes:
#  - VADER must be installed before running.  The pipeline will raise a
#    RuntimeError if vaderSentiment is absent (silent fallback disabled).
#  - A reproducibility manifest is written to results/ on every run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
ANNOMI_DIR="${ANNOMI_DIR:-${ROOT_DIR}/data/annomi}"
RFS_OUT_v16="${RFS_OUT_v16:-${ROOT_DIR}/results}"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --annomi-dir)
      ANNOMI_DIR="$2"; shift 2 ;;
    --output-dir)
      RFS_OUT_v16="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── pre-flight checks ────────────────────────────────────────────────────────
echo "============================================================"
echo "  RFS-SCP v16.1 — Full Pipeline"
echo "  AnnoMI dir : ${ANNOMI_DIR}"
echo "  Output dir : ${RFS_OUT_v16}"
echo "============================================================"

if [[ ! -f "${ANNOMI_DIR}/AnnoMI-full.csv" ]] && \
   [[ ! -f "${ANNOMI_DIR}/AnnoMI-simple.csv" ]]; then
  echo "[ERROR] AnnoMI CSV not found in ${ANNOMI_DIR}"
  echo "  Download from https://github.com/uccollab/annomi"
  echo "  and place at ${ANNOMI_DIR}/AnnoMI-full.csv"
  exit 1
fi

python -c "import vaderSentiment" 2>/dev/null || {
  echo "[ERROR] vaderSentiment not found. pip install vaderSentiment"
  exit 1
}

# ── run main script ───────────────────────────────────────────────────────────
export RFS_OUT_v16

python "${ROOT_DIR}/src/rfs_scp_v16_main.py" \
  --annomi-dir "${ANNOMI_DIR}" \
  "$@"

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Results → ${RFS_OUT_v16}"
echo "============================================================"
