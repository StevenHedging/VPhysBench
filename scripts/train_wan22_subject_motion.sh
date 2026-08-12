#!/usr/bin/env bash
set -euo pipefail

: "${WAN_PROJECT_ROOT:?WAN_PROJECT_ROOT is required}"
: "${WAN_PYTHON:?WAN_PYTHON is required}"
: "${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${METADATA_PATH:?METADATA_PATH is required}"
: "${MODEL_BASE:?MODEL_BASE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SUBJECT_MOTION_CONFIG_JSON:?SUBJECT_MOTION_CONFIG_JSON is required}"
: "${SUBJECT_MOTION_METRICS_PATH:?SUBJECT_MOTION_METRICS_PATH is required}"
: "${ACCELERATE_CONFIG:?ACCELERATE_CONFIG is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
: "${LORA_CHECKPOINT:?LORA_CHECKPOINT is required}"

if [[ "$CUDA_VISIBLE_DEVICES" != "0,1,2,3" ]]; then
  echo "Subject-motion training requires CUDA_VISIBLE_DEVICES=0,1,2,3." >&2
  exit 1
fi

DS="$WAN_PROJECT_ROOT/vendor/DiffSynth-Studio"
MODEL_DIR="$MODEL_BASE/Wan-AI/Wan2.2-TI2V-5B"
MODEL_PATHS="[\"$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth\",[\"$MODEL_DIR/diffusion_pytorch_model-00001-of-00003.safetensors\",\"$MODEL_DIR/diffusion_pytorch_model-00002-of-00003.safetensors\",\"$MODEL_DIR/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"$MODEL_DIR/Wan2.2_VAE.pth\"]"
PYTHON_REAL="$(readlink -f "$WAN_PYTHON")"
PYTHON_ENV_ROOT="$(cd "$(dirname "$PYTHON_REAL")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-$PYTHON_ENV_ROOT}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BENCHMARK_ROOT/src:$BENCHMARK_ROOT:$DS:${PYTHONPATH:-}"

HEIGHT="${HEIGHT:-832}"
WIDTH="${WIDTH:-480}"
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MAX_PIXELS="${MAX_PIXELS:-399360}"
NUM_FRAMES="${NUM_FRAMES:-121}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-1}"

if [[ "$LORA_RANK" != "32" || "$LORA_TARGET_MODULES" != "q,k,v,o,ffn.0,ffn.2" ]]; then
  echo "Subject-motion training requires sealed rank-32 LoRA targets." >&2
  exit 1
fi

PAIRED_HEAD="${LORA_CHECKPOINT%.safetensors}.st-head.safetensors"
for required in \
  "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  "$MODEL_DIR/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "$MODEL_DIR/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "$MODEL_DIR/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "$MODEL_DIR/Wan2.2_VAE.pth" \
  "$MODEL_DIR/google/umt5-xxl/tokenizer.json" \
  "$METADATA_PATH" \
  "$ACCELERATE_CONFIG" \
  "$LORA_CHECKPOINT" \
  "$PAIRED_HEAD"; do
  if [[ ! -f "$required" ]]; then
    echo "Required subject-motion training asset is missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
if compgen -G "$OUTPUT_DIR/*.safetensors" >/dev/null && \
   [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "Refusing to reuse a checkpoint directory: $OUTPUT_DIR" >&2
  exit 1
fi

{
  echo "run_name=${RUN_NAME:-$(basename "$OUTPUT_DIR")}" 
  echo "metadata_path=$METADATA_PATH"
  echo "num_frames=$NUM_FRAMES"
  echo "dataset_repeat=$DATASET_REPEAT"
  echo "num_epochs=$NUM_EPOCHS"
  echo "learning_rate=$LEARNING_RATE"
  echo "weight_decay=$WEIGHT_DECAY"
  echo "lora_rank=$LORA_RANK"
  echo "lora_target_modules=$LORA_TARGET_MODULES"
  echo "gradient_accumulation_steps=$GRAD_ACCUM"
  echo "train_seed=$TRAIN_SEED"
  echo "save_steps=${SAVE_STEPS:-none}"
  echo "lora_checkpoint=$LORA_CHECKPOINT"
  echo "paired_head=$PAIRED_HEAD"
  echo "save_optimizer_state=$SAVE_OPTIMIZER_STATE"
  echo "subject_motion_config_json=$SUBJECT_MOTION_CONFIG_JSON"
  echo "subject_motion_metrics_path=$SUBJECT_MOTION_METRICS_PATH"
  echo "accelerate_config=$ACCELERATE_CONFIG"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} > "$OUTPUT_DIR/run.env"

extra_args=(--lora_checkpoint "$LORA_CHECKPOINT")
if [[ -n "$SAVE_STEPS" ]]; then
  extra_args+=(--save_steps "$SAVE_STEPS")
fi
if [[ "$SAVE_OPTIMIZER_STATE" == "1" ]]; then
  extra_args+=(--save_optimizer_state)
fi

size_args=()
if [[ "$DYNAMIC_RESOLUTION" == "1" ]]; then
  size_args+=(--max_pixels "$MAX_PIXELS")
else
  size_args+=(--height "$HEIGHT" --width "$WIDTH")
fi

cd "$DS"
"$WAN_PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  "$BENCHMARK_ROOT/scripts/wan22_subject_motion_train.py" \
  --dataset_base_path "$DATA_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys video,subject_mask \
  "${size_args[@]}" \
  --num_frames "$NUM_FRAMES" \
  --dataset_repeat "$DATASET_REPEAT" \
  --dataset_num_workers "$NUM_WORKERS" \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$MODEL_DIR/google/umt5-xxl" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_epochs "$NUM_EPOCHS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --use_gradient_checkpointing \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model dit \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank "$LORA_RANK" \
  --extra_inputs input_image \
  --enable_tensorboard_log \
  --subject_motion_config_json "$SUBJECT_MOTION_CONFIG_JSON" \
  --subject_motion_metrics_path "$SUBJECT_MOTION_METRICS_PATH" \
  --sampler_seed "$TRAIN_SEED" \
  "${extra_args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
