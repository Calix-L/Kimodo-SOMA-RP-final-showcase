#!/usr/bin/env bash
# End-to-end Kimodo generation and evaluation pipeline.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash inference/run_pipeline.sh /path/to/run.env" >&2
  exit 2
fi

CONFIG_FILE="$1"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file does not exist: $CONFIG_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:?Set PYTHON in the config file}"
KIMODO_ROOT="${KIMODO_ROOT:?Set KIMODO_ROOT in the config file}"
BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR in the config file}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR in the config file}"
MODEL_NAME="${MODEL_NAME:-Kimodo-SOMA-RP-v1.1}"
MODEL_LABEL="${MODEL_LABEL:-rp_base}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT in the config file}"
TEXT_ENCODERS_DIR="${TEXT_ENCODERS_DIR:?Set TEXT_ENCODERS_DIR in the config file}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
EXPECTED_COUNT="${EXPECTED_COUNT:-}"
POSTPROCESS="${POSTPROCESS:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODER_MODE=local
export TEXT_ENCODERS_DIR
export CHECKPOINT_DIR="$CHECKPOINT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/generated" "$RUN_DIR/eval_input" "$RUN_DIR/metrics"

generate_args=(
  "$SCRIPT_DIR/scripts/generate.py"
  --kimodo-root "$KIMODO_ROOT"
)
if [[ -n "$CHECKPOINT" ]]; then
  generate_args+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${FEATURE_INDEX:-}" || -n "${TEXT_FEATURE_DIR:-}" ]]; then
  generate_args+=(--feature-index "${FEATURE_INDEX:?}" --text-feature-dir "${TEXT_FEATURE_DIR:?}")
fi
generate_args+=(
  --benchmark "$BENCHMARK_DIR"
  --output "$RUN_DIR/generated"
  --model "$MODEL_NAME"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --diffusion_steps "$DIFFUSION_STEPS"
  --text_encoder_fp32
)
if [[ "$POSTPROCESS" == "1" ]]; then
  generate_args+=(--postprocess)
fi

echo "[1/5] Generating motions"
"$PYTHON" "${generate_args[@]}" 2>&1 | tee "$RUN_DIR/logs/generate.log"

manifest_args=(
  "$SCRIPT_DIR/scripts/build_manifest.py"
  --generated-root "$RUN_DIR/generated"
  --source-manifest "$BENCHMARK_DIR/manifest.json"
  --output-dir "$RUN_DIR/eval_input"
  --model "$MODEL_LABEL"
  --checkpoint "$CHECKPOINT"
  --diffusion-steps "$DIFFUSION_STEPS"
)
if [[ "$POSTPROCESS" == "1" ]]; then
  manifest_args+=(--postprocess)
fi
if [[ -n "$EXPECTED_COUNT" ]]; then
  manifest_args+=(--expected-count "$EXPECTED_COUNT")
fi

echo "[2/5] Building the canonical evaluation manifest"
"$PYTHON" "${manifest_args[@]}" 2>&1 | tee "$RUN_DIR/logs/build_manifest.log"

validate_args=(
  "$SCRIPT_DIR/scripts/validate_outputs.py"
  --manifest "$RUN_DIR/eval_input/manifest.csv"
)
if [[ -n "$EXPECTED_COUNT" ]]; then
  validate_args+=(--expected-count "$EXPECTED_COUNT")
fi

echo "[3/5] Validating all generated NPZ files"
"$PYTHON" "${validate_args[@]}" 2>&1 | tee "$RUN_DIR/logs/validate.log"

echo "[4/5] Computing TMR and physical metrics"
"$PYTHON" "$SCRIPT_DIR/scripts/evaluate_tmr_physics.py" \
  --kimodo-root "$KIMODO_ROOT" \
  --manifest "$RUN_DIR/eval_input/manifest.csv" \
  --output-dir "$RUN_DIR/metrics/tmr_physical" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --text-encoders-dir "$TEXT_ENCODERS_DIR" \
  --device "$DEVICE" \
  --physical-batch-size "${PHYSICAL_BATCH_SIZE:-32}" \
  --motion-batch-size "${MOTION_BATCH_SIZE:-1}" \
  --text-batch-size "${TEXT_BATCH_SIZE:-32}" \
  --tmr-max-frames "${TMR_MAX_FRAMES:-300}" \
  2>&1 | tee "$RUN_DIR/logs/evaluate_tmr_physical.log"

if [[ -n "${GT_FEATURE_INDEX:-}" || -n "${GT_FEATURE_CACHE:-}" ]]; then
  echo "[5/5] Computing paired RTE/BPE"
  "$PYTHON" "$SCRIPT_DIR/scripts/evaluate_rte_bpe.py" \
    --kimodo-root "$KIMODO_ROOT" \
    --repro-root "${REPRO_ROOT:?Set REPRO_ROOT when RTE/BPE is enabled}" \
    --feature-index "${GT_FEATURE_INDEX:?}" \
    --feature-cache "${GT_FEATURE_CACHE:?}" \
    --prediction "$MODEL_LABEL=$RUN_DIR/eval_input/manifest.csv" \
    --output-dir "$RUN_DIR/metrics/rte_bpe" \
    --model "$MODEL_NAME" \
    --device "$DEVICE" \
    2>&1 | tee "$RUN_DIR/logs/evaluate_rte_bpe.log"
else
  echo "[5/5] Skipping RTE/BPE because no paired Game GT was configured"
fi

echo "Pipeline complete: $RUN_DIR"
