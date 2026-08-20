#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/share/user/zyc/kimodo-overfit
KIMODO_ROOT=/home/share/user/zyc/kimodo
DATASET_ROOT=/home/share/user/zyc/kimodo_phase2/datasets/kimodo_retarget_v6_human7951_20260812
FEATURE_CACHE="$ROOT/work/feature_cache/human7951"
RUN_DIR="$ROOT/work/runs/human7951_overfit"
PY=${PY:-/opt/conda/bin/python}

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_ENCODER_MODE=local
export TEXT_ENCODERS_DIR=/home/share/user/zyc/kimodo/text_encoders
export PYTHONPATH="$KIMODO_ROOT:$ROOT:${PYTHONPATH:-}"

mkdir -p "$ROOT/work/logs" "$ROOT/work/runs" "$ROOT/work/feature_cache"

if [ ! -f "$FEATURE_CACHE/index.jsonl" ]; then
  "$PY" "$ROOT/scripts/build_human7951_feature_cache.py" \
    --dataset-root "$DATASET_ROOT" \
    --out-cache "$FEATURE_CACHE"
fi

if [ ! -f "$ROOT/work/feature_cache/human7951_text/text_cache_manifest.json" ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "$PY" "$ROOT/scripts/cache_text_features.py" \
    --kimodo-root "$KIMODO_ROOT" \
    --feature-cache "$FEATURE_CACHE" \
    --out-dir "$ROOT/work/feature_cache/human7951_text" \
    --model Kimodo-SOMA-RP-v1.1 \
    --batch-size 32
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "$PY" "$ROOT/train.py" \
  --config "$ROOT/configs/overfit_human7951.yaml" \
  --kimodo-root "$KIMODO_ROOT" \
  --feature-cache "$FEATURE_CACHE" \
  --output-dir "$RUN_DIR"

"$PY" "$ROOT/scripts/plot_loss.py" --run-dir "$RUN_DIR"
