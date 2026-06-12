#!/usr/bin/env bash
# =============================================================================
#  run_eda.sh — AnnoMI Dataset EDA (Exploratory Data Analysis)
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Usage
#  -----
#    bash scripts/run_eda.sh [--annomi-dir PATH]
#
#  Environment variables (can be set before calling this script)
#    ANNOMI_DIR   : path to directory containing AnnoMI-full.csv (or simple)
#                   default: data/annomi
#    RFS_OUT_V12  : output directory for EDA artifacts
#                   default: rfs_v12_outputs
#
#  Outputs
#  -------
#    annomi_eda_summary.csv        : per-column stats
#    annomi_session_features_v12.csv : extracted session features
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Parse optional args
# ---------------------------------------------------------------------------

ANNOMI_DIR="${ANNOMI_DIR:-data/annomi}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --annomi-dir)
            ANNOMI_DIR="$2"; shift 2 ;;
        --annomi-dir=*)
            ANNOMI_DIR="${1#*=}"; shift ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
#  Validate environment
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  RFS-SCP v12.0  —  EDA Pipeline"
echo "============================================================"
echo "  ANNOMI_DIR : $ANNOMI_DIR"

if [[ ! -f "$ANNOMI_DIR/AnnoMI-full.csv" ]] && \
   [[ ! -f "$ANNOMI_DIR/AnnoMI-simple.csv" ]]; then
    echo ""
    echo "  ERROR: No AnnoMI CSV found in '$ANNOMI_DIR'."
    echo ""
    echo "  Please download the dataset and place it at:"
    echo "    $ANNOMI_DIR/AnnoMI-full.csv"
    echo ""
    echo "  Dataset source:"
    echo "    https://github.com/uccollab/annomi"
    echo ""
    echo "  Citation: Wu et al. (2022). AnnoMI: A Dataset of"
    echo "    Expert-Annotated Counselling Dialogues."
    exit 1
fi

# ---------------------------------------------------------------------------
#  Check Python dependencies
# ---------------------------------------------------------------------------

echo ""
echo "  Checking Python dependencies ..."
python3 -c "import numpy, pandas, matplotlib, scipy, sklearn" 2>/dev/null || {
    echo ""
    echo "  ERROR: Missing Python dependencies."
    echo "  Install with: pip install -r requirements.txt"
    exit 1
}

# ---------------------------------------------------------------------------
#  Run EDA
# ---------------------------------------------------------------------------

echo ""
echo "  Running AnnoMI EDA ..."
echo ""

ANNOMI_DIR="$ANNOMI_DIR" python3 src/eda/annomi_eda.py \
    --annomi-dir "$ANNOMI_DIR"

echo ""
echo "============================================================"
echo "  EDA complete."
echo "  Summary saved to: annomi_eda_summary.csv"
echo "============================================================"
