#!/usr/bin/env bash
# =============================================================================
#  run_full_pipeline.sh — Full RFS-SCP v12.0 Analysis Pipeline
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Usage
#  -----
#    bash scripts/run_full_pipeline.sh [OPTIONS]
#
#  Options
#    --annomi-dir PATH     path to AnnoMI CSVs (default: data/annomi)
#    --output-dir PATH     output directory    (default: rfs_v12_outputs)
#    --skip-sllm           skip SLLM benchmark (faster run, no Ollama needed)
#    --ollama-url URL      Ollama server URL   (default: http://localhost:11434)
#    --gpu                 hint: use GPU for LSTM/BiLSTM (auto-detected)
#    -h | --help           show this help
#
#  Environment variables
#    ANNOMI_DIR            overrides --annomi-dir
#    RFS_OUT_V12           overrides --output-dir
#    OLLAMA_URL            overrides --ollama-url
#
#  Estimated runtime
#    CPU only  : ~2–4 hours  (LSTM sensitivity grid is the bottleneck)
#    GPU (RTX3060+): ~20 minutes
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------

ANNOMI_DIR="${ANNOMI_DIR:-data/annomi}"
OUTPUT_DIR="${RFS_OUT_V12:-rfs_v12_outputs}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/generate}"
SKIP_SLLM=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/pipeline_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
#  Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --annomi-dir)   ANNOMI_DIR="$2";  shift 2 ;;
        --annomi-dir=*) ANNOMI_DIR="${1#*=}"; shift ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --ollama-url)   OLLAMA_URL="$2";  shift 2 ;;
        --ollama-url=*) OLLAMA_URL="${1#*=}"; shift ;;
        --skip-sllm)    SKIP_SLLM=true;   shift ;;
        --gpu)          export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"; shift ;;
        -h|--help)
            grep "^#" "$0" | head -30 | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
#  Setup
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  RFS-SCP v12.0 — Full Analysis Pipeline"
echo "  Start : $(date)"
echo "============================================================"
echo "  ANNOMI_DIR : $ANNOMI_DIR"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  SKIP_SLLM  : $SKIP_SLLM"
echo "  LOG        : $LOG_FILE"
echo ""

# ---------------------------------------------------------------------------
#  Dependency checks
# ---------------------------------------------------------------------------

echo "  [0/5] Checking dependencies ..."

python3 -c "import numpy, pandas, matplotlib, scipy, sklearn, statsmodels" \
    || { echo "ERROR: core Python deps missing. pip install -r requirements.txt"; exit 1; }

python3 -c "import torch" 2>/dev/null \
    && echo "       PyTorch   : OK" \
    || echo "       PyTorch   : NOT FOUND — LSTM→Ridge fallback"

python3 -c "import imblearn" 2>/dev/null \
    && echo "       imbalanced-learn : OK" \
    || echo "       imbalanced-learn : NOT FOUND — SMOTE skipped"

python3 -c "import vaderSentiment" 2>/dev/null \
    && echo "       VADER     : OK" \
    || echo "       VADER     : NOT FOUND — lexical fallback"

echo ""

# ---------------------------------------------------------------------------
#  Data validation
# ---------------------------------------------------------------------------

echo "  [1/5] Validating AnnoMI data ..."

if [[ -f "$ANNOMI_DIR/AnnoMI-full.csv" ]]; then
    N_ROWS=$(python3 -c "import pandas as pd; df=pd.read_csv('$ANNOMI_DIR/AnnoMI-full.csv'); print(len(df))")
    echo "       AnnoMI-full.csv: $N_ROWS utterances"
elif [[ -f "$ANNOMI_DIR/AnnoMI-simple.csv" ]]; then
    N_ROWS=$(python3 -c "import pandas as pd; df=pd.read_csv('$ANNOMI_DIR/AnnoMI-simple.csv'); print(len(df))")
    echo "       AnnoMI-simple.csv (fallback): $N_ROWS utterances"
    echo "       WARNING: Results may differ from v12 paper (full dataset preferred)"
else
    echo ""
    echo "  ERROR: No AnnoMI CSV found in '$ANNOMI_DIR'."
    echo "  Download from: https://github.com/uccollab/annomi"
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
#  Ollama check (unless skipping SLLM)
# ---------------------------------------------------------------------------

if [[ "$SKIP_SLLM" == "false" ]]; then
    echo "  [2/5] Checking Ollama SLLM server ..."
    if curl -sf "${OLLAMA_URL%/api/generate}/api/tags" > /dev/null 2>&1; then
        echo "       Ollama server : reachable"
        LOADED=$(curl -sf "${OLLAMA_URL%/api/generate}/api/tags" \
                 | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['name'] for m in d.get('models',[])])" 2>/dev/null || echo "[]")
        echo "       Loaded models : $LOADED"
    else
        echo "       Ollama server : NOT REACHABLE"
        echo "       SLLM benchmark will use lexical fallback estimator"
        OLLAMA_URL="http://not-available.local/api/generate"
    fi
    echo ""
else
    echo "  [2/5] Skipping SLLM check (--skip-sllm)"
    OLLAMA_URL="http://not-available.local/api/generate"
    echo ""
fi

# ---------------------------------------------------------------------------
#  Run main pipeline
# ---------------------------------------------------------------------------

echo "  [3/5] Running RFS-SCP v12 main pipeline ..."
echo "        (This may take 20 min – 4 hours depending on hardware)"
echo ""

ANNOMI_DIR="$ANNOMI_DIR" \
RFS_OUT_V12="$OUTPUT_DIR" \
OLLAMA_URL="$OLLAMA_URL" \
python3 src/rfs_scp_v12_main.py \
    --annomi-dir "$ANNOMI_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ollama-url "$OLLAMA_URL"

echo ""

# ---------------------------------------------------------------------------
#  Verify outputs
# ---------------------------------------------------------------------------

echo "  [4/5] Verifying output files ..."

EXPECTED_FILES=(
    "annomi_session_features_v12.csv"
    "hypothesis_summary_v12.csv"
    "shap_linear_v12.csv"
    "shap_permutation_v12.csv"
    "shap_gsshap_v12.csv"
    "sllm_benchmark_v12.csv"
    "rfs_controller_log_v12.csv"
    "fig1_rq1_external_v12.png"
    "fig2_rq2_moderation_v12.png"
    "fig3_rq4_bayesian_v12.png"
    "fig4_shap_v12.png"
    "fig5_ablation_v12.png"
    "fig6_rq3_cluster_v12.png"
    "fig7_sllm_v12.png"
    "fig8_dynamics_umap_v12.png"
    "fig9_power_scorecard_v12.png"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$OUTPUT_DIR/$f" ]]; then
        SIZE=$(du -h "$OUTPUT_DIR/$f" | cut -f1)
        printf "       %-55s %s\n" "$f" "$SIZE"
    else
        echo "       MISSING: $f"
        MISSING=$((MISSING+1))
    fi
done

echo ""

# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

echo "  [5/5] Pipeline summary"
echo ""

if [[ $MISSING -eq 0 ]]; then
    echo "       All expected output files generated successfully."
else
    echo "       WARNING: $MISSING file(s) missing from expected outputs."
fi

echo ""
echo "============================================================"
echo "  RFS-SCP v12.0 — Complete"
echo "  End   : $(date)"
echo "  Output: $OUTPUT_DIR"
echo "  Log   : $LOG_FILE"
echo "============================================================"
