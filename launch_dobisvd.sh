#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   MODEL_ID=Qwen/Qwen3-8B TARGET_RATIO=0.2 SKIP_EVAL=1 CUDA_VISIBLE_DEVICES=0 ./launch_dobisvd.sh
#   MODEL_ID=mistralai/Mistral-7B-v0.1 TARGET_RATIO=0.2 SKIP_EVAL=1 CUDA_VISIBLE_DEVICES=0 ./launch_dobisvd.sh

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
TARGET_RATIO="${TARGET_RATIO:-0.2}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-0}"
TRAINING_DATASET="${TRAINING_DATASET:-wikitext2_evol_codealpaca_tulu_math}"
N_TRAIN_EPOCHS="${N_TRAIN_EPOCHS:-20}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-256}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-256}"
PATH_HEAD_FOLDER="${PATH_HEAD_FOLDER:-./}"
PATH_HEAD_FOLDER_OUTPUT="${PATH_HEAD_FOLDER_OUTPUT:-./results}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
REMAPPING="${REMAPPING:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"

EXTRA_FLAGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  EXTRA_FLAGS+=(--trust_remote_code)
fi
if [[ "${SKIP_EVAL}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_eval)
fi
if [[ "${REMAPPING}" == "1" ]]; then
  EXTRA_FLAGS+=(--remapping)
  RESULT_PREFIX="Diff-Remapping"
else
  RESULT_PREFIX="Diff-Noremapping"
fi

python svd_trainer.py \
  --model_id "${MODEL_ID}" \
  --target_ratio "${TARGET_RATIO}" \
  --seq_len "${SEQ_LEN}" \
  --seed "${SEED}" \
  --training_dataset "${TRAINING_DATASET}" \
  --n_train_epochs "${N_TRAIN_EPOCHS}" \
  --n_train_samples "${N_TRAIN_SAMPLES}" \
  --n_eval_samples "${N_EVAL_SAMPLES}" \
  --path_head_folder "${PATH_HEAD_FOLDER}" \
  --path_head_folder_output "${PATH_HEAD_FOLDER_OUTPUT}" \
  "${EXTRA_FLAGS[@]}"

MODEL_NAME="${MODEL_ID##*/}"
TRAINING_OUTPUT_DIR="${PATH_HEAD_FOLDER_OUTPUT%/}/training_output/${MODEL_NAME}"
TRAINING_RESULT_PATH="$(find "${TRAINING_OUTPUT_DIR}" -maxdepth 1 -type d -name "${RESULT_PREFIX}-${TARGET_RATIO}_${TRAINING_DATASET}_${SEQ_LEN}_*" | sort | tail -n 1)"

if [[ -z "${TRAINING_RESULT_PATH}" ]]; then
  echo "Could not find training result under ${TRAINING_OUTPUT_DIR}" >&2
  exit 1
fi

UPDATE_FLAGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  UPDATE_FLAGS+=(--trust_remote_code)
fi
if [[ "${REMAPPING}" == "1" ]]; then
  UPDATE_FLAGS+=(--remapping)
fi

python weight_updater.py \
  --model_id "${MODEL_ID}" \
  --training_result_path "$(basename "${TRAINING_RESULT_PATH}")" \
  --seed "${SEED}" \
  --training_dataset "${TRAINING_DATASET}" \
  --n_train_samples "${N_TRAIN_SAMPLES}" \
  --n_eval_samples "${N_EVAL_SAMPLES}" \
  --path_head_folder "${PATH_HEAD_FOLDER}" \
  --path_head_folder_output "${PATH_HEAD_FOLDER_OUTPUT}" \
  "${UPDATE_FLAGS[@]}"
