#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/benchmark/neural_networks"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
CHECKPOINT="${REPO_ROOT}/checkpoints/large_weights_only_196.pth"
DATASET_YAML="${REPO_ROOT}/easy_start/configs/bci_iv2a_dataset_specs.yaml"
DOWNSTREAM_TASK_YAML="${REPO_ROOT}/easy_start/configs/bci_iv2a_downstream_task_specs.yaml"
OUTPUT_DIR="${REPO_ROOT}/output_dir/bci_iv2a_official_like"
LOG_DIR="${REPO_ROOT}/runs/bci_iv2a_official_like"
WANDB_LOG_DIR="${REPO_ROOT}/wandb_logs/bci_iv2a_official_like"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_YAML}" ]]; then
  echo "Missing dataset YAML: ${DATASET_YAML}" >&2
  exit 1
fi

if [[ ! -f "${DOWNSTREAM_TASK_YAML}" ]]; then
  echo "Missing downstream task YAML: ${DOWNSTREAM_TASK_YAML}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${WANDB_LOG_DIR}"

cd "${BENCHMARK_DIR}"

exec "${PYTHON_BIN}" wandb_downstream_evaluation.py \
  --downstream_task bci_iv2a \
  --evaluation_scheme per-subject \
  --model vit_large_patch16 \
  --vit_pretrained_model_dir "${CHECKPOINT}" \
  --optimizer_spec finetune \
  --train_epochs 100 \
  --finetune_epochs 50 \
  --lr 1e-3 \
  --min_lr 1e-6 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.0 \
  --proj_drop_rate 0.0 \
  --train_drop_rate 0.0 \
  --train_attn_drop_rate 0.0 \
  --train_proj_drop_rate 0.0 \
  --finetune_drop_rate 0.0 \
  --finetune_attn_drop_rate 0.0 \
  --finetune_proj_drop_rate 0.0 \
  --train_warmup_epochs 10 \
  --finetune_warmup_epochs 10 \
  --mix_up 0.9 \
  --smoothing 0.1 \
  --dataset_yaml "${DATASET_YAML}" \
  --downstream_task_yaml "${DOWNSTREAM_TASK_YAML}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --wandb_log_dir "${WANDB_LOG_DIR}" \
  --wandb_project bci_iv2a_official_like \
  "$@"
