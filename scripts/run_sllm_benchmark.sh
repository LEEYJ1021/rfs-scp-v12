#!/usr/bin/env bash
# =============================================================================
#  run_sllm_benchmark.sh — Multi-SLLM Benchmark via Ollama
#
#  Part of: RFS-SCP v12.0 (Circumplex-Grounded Relational State Estimation)
#
#  Usage
#  -----
#    bash scripts/run_sllm_benchmark.sh [OPTIONS]
#
#  Options
#    --annomi-dir PATH    path to AnnoMI CSVs  (default: data/annomi)
#    --output-dir PATH    output directory      (default: rfs_v12_outputs)
#    --ollama-url URL     Ollama API endpoint
#                         (default: http://localhost:11434/api/generate)
#    --models "m1 m2 .."  space-separated Ollama model tags to benchmark
#    --pull               pull all listed models before running
#    -h | --help          show this help
#
#  Prerequisites
#  -------------
#    - Ollama installed and running: https://ollama.ai/download
#      Start server: ollama serve &
#    - Minimum RAM: 16 GB (32 GB recommended for 7B models)
#
#  Default models benchmarked (v12)
#    qwen2.5:1.5b  qwen2.5:3b  qwen2.5:7b  phi3:mini
#    gemma2:2b  llama3.2:3b  mistral:7b-instruct
#
#  Outputs
#    sllm_benchmark_v12.csv  — per-model ICC, AUC, AP, parse_rate
#    sllm_labels_v12.csv     — per-session labels from each model
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------

ANNOMI_DIR="${ANNOMI_DIR:-data/annomi}"
OUTPUT_DIR="${RFS_OUT_V12:-rfs_v12_outputs}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/generate}"
PULL_MODELS=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DEFAULT_MODELS=(
    "qwen2.5:1.5b"
    "qwen2.5:3b"
    "qwen2.5:7b"
    "phi3:mini"
    "gemma2:2b"
    "llama3.2:3b"
    "mistral:7b-instruct"
)
MODELS=()

# ---------------------------------------------------------------------------
#  Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --annomi-dir)    ANNOMI_DIR="$2"; shift 2 ;;
        --annomi-dir=*)  ANNOMI_DIR="${1#*=}"; shift ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*)  OUTPUT_DIR="${1#*=}"; shift ;;
        --ollama-url)    OLLAMA_URL="$2"; shift 2 ;;
        --ollama-url=*)  OLLAMA_URL="${1#*=}"; shift ;;
        --models)
            IFS=' ' read -ra MODELS <<< "$2"; shift 2 ;;
        --pull)          PULL_MODELS=true; shift ;;
        -h|--help)
            grep "^#" "$0" | head -40 | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${DEFAULT_MODELS[@]}")

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/sllm_benchmark_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  RFS-SCP v12.0 — Multi-SLLM Benchmark  [V12-C]"
echo "  Start : $(date)"
echo "============================================================"
echo "  ANNOMI_DIR : $ANNOMI_DIR"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  OLLAMA_URL : $OLLAMA_URL"
echo "  MODELS     : ${MODELS[*]}"
echo "  LOG        : $LOG_FILE"
echo ""

# ---------------------------------------------------------------------------
#  Validate Ollama server
# ---------------------------------------------------------------------------

TAGS_URL="${OLLAMA_URL%/api/generate}/api/tags"

echo "  Checking Ollama server at ${TAGS_URL} ..."
if ! curl -sf "$TAGS_URL" > /dev/null 2>&1; then
    echo ""
    echo "  ERROR: Ollama server not reachable."
    echo ""
    echo "  To start Ollama:"
    echo "    ollama serve &"
    echo ""
    echo "  To install Ollama:"
    echo "    https://ollama.ai/download"
    echo ""
    echo "  Alternatively, re-run with fallback only:"
    echo "    OLLAMA_URL=http://not-available.local/api/generate \\"
    echo "      bash scripts/run_sllm_benchmark.sh"
    exit 1
fi

echo "  Ollama server: reachable"

# ---------------------------------------------------------------------------
#  Pull models (optional)
# ---------------------------------------------------------------------------

if [[ "$PULL_MODELS" == "true" ]]; then
    echo ""
    echo "  Pulling models ..."
    for model in "${MODELS[@]}"; do
        echo "    ollama pull $model"
        ollama pull "$model" || echo "    WARNING: Failed to pull $model"
    done
fi

# ---------------------------------------------------------------------------
#  RAM check (informational)
# ---------------------------------------------------------------------------

TOTAL_RAM_GB=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['free', '-g'], text=True)
    m = re.search(r'Mem:\s+(\d+)', out)
    print(int(m.group(1)) if m else '?')
except Exception:
    print('?')
")
echo "  System RAM: ${TOTAL_RAM_GB} GB"
if [[ "$TOTAL_RAM_GB" != "?" ]] && [[ "$TOTAL_RAM_GB" -lt 16 ]]; then
    echo "  WARNING: < 16 GB RAM detected. 7B models may fail or be very slow."
fi

echo ""

# ---------------------------------------------------------------------------
#  Run benchmark
# ---------------------------------------------------------------------------

echo "  Running SLLM benchmark ..."
echo "  (Each session: 1 Ollama call per model × 133 sessions)"
echo ""

ANNOMI_DIR="$ANNOMI_DIR" \
RFS_OUT_V12="$OUTPUT_DIR" \
OLLAMA_URL="$OLLAMA_URL" \
python3 src/sllm/sllm_benchmark.py \
    --annomi-dir "$ANNOMI_DIR" \
    --ollama-url "$OLLAMA_URL" \
    --ollama-models "${MODELS[@]}"

echo ""

# ---------------------------------------------------------------------------
#  Post-processing: print summary table
# ---------------------------------------------------------------------------

echo "  SLLM Benchmark Results:"
echo ""

python3 - <<'PYEOF'
import pandas as pd, sys, os

csv_path = os.path.join(os.environ.get("RFS_OUT_V12", "rfs_v12_outputs"),
                         "sllm_benchmark_v12.csv")
if not os.path.exists(csv_path):
    print("  (results CSV not found)")
    sys.exit(0)

df = pd.read_csv(csv_path)
cols = ["model", "params_b", "ICC_cohesion", "Spearman_rho", "AUC", "AP",
        "parse_rate", "latency_ms"]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False, float_format="{:.3f}".format))
print()
print(f"  Best model: {df.loc[df['AUC'].idxmax(), 'model']}  "
      f"AUC={df['AUC'].max():.4f}")
print(f"  ICC max   : {df['ICC_cohesion'].max():.3f}  "
      f"(threshold for 'fair agreement' = 0.40)")
PYEOF

echo ""
echo "============================================================"
echo "  SLLM Benchmark complete."
echo "  End   : $(date)"
echo "  Output: $OUTPUT_DIR/sllm_benchmark_v12.csv"
echo "  Log   : $LOG_FILE"
echo "============================================================"
