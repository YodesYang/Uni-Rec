#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python3 -u "${SCRIPT_DIR}/eda_hour_weight_audit.py" \
    --data_dir "${TRAIN_DATA_PATH:-/data_ams/industrial_training_data}" \
    --temporal_weight_alpha 2.5 \
    --hour_weight_min 9 \
    --hour_weight_max 14 \
    --hour_weight_multiplier 5.0 \
    --target_date 2026-03-04 \
    --max_row_groups 0
