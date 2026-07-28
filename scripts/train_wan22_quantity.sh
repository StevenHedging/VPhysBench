#!/usr/bin/env bash
set -euo pipefail

: "${WAN_PROJECT_ROOT:?WAN_PROJECT_ROOT is required}"
: "${WAN_PYTHON:?WAN_PYTHON is required}"
: "${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${METADATA_PATH:?METADATA_PATH is required}"
: "${MODEL_BASE:?MODEL_BASE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${QUANTITY_ENCODER_CONFIG_JSON:?QUANTITY_ENCODER_CONFIG_JSON is required}"
: "${QUANTITY_TOKEN_AUDIT_PATH:?QUANTITY_TOKEN_AUDIT_PATH is required}"

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
export PYTHONPATH="$BENCHMARK_ROOT/src:$DS:${PYTHONPATH:-}"

HEIGHT="${HEIGHT:-832}"
WIDTH="${WIDTH:-480}"
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MAX_PIXELS="${MAX_PIXELS:-399360}"
NUM_FRAMES="${NUM_FRAMES:-121}"
DATASET_REPEAT="${DATASET_REPEAT:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:?ACCELERATE_CONFIG is required}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-1}"

if [[ -n "$LORA_CHECKPOINT" ]]; then
  echo "Combined-checkpoint resume is not supported by this Baseline version." >&2
  echo "Use a fresh run with random DiT-LoRA and quantity-encoder initialization." >&2
  exit 1
fi

for required in \
  "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  "$MODEL_DIR/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "$MODEL_DIR/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "$MODEL_DIR/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "$MODEL_DIR/Wan2.2_VAE.pth" \
  "$MODEL_DIR/google/umt5-xxl/tokenizer.json" \
  "$METADATA_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "Required training asset is missing: $required" >&2
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
  echo "dynamic_resolution=$DYNAMIC_RESOLUTION"
  echo "max_pixels=$MAX_PIXELS"
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
  echo "lora_checkpoint=${LORA_CHECKPOINT:-none}"
  echo "save_optimizer_state=$SAVE_OPTIMIZER_STATE"
  echo "accelerate_config=$ACCELERATE_CONFIG"
} > "$OUTPUT_DIR/run.env"

extra_args=()
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
  "$BENCHMARK_ROOT/scripts/wan22_quantity_train.py" \
  --dataset_base_path "$DATA_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys video \
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
  --quantity_encoder_config_json "$QUANTITY_ENCODER_CONFIG_JSON" \
  --quantity_token_audit_path "$QUANTITY_TOKEN_AUDIT_PATH" \
  "${extra_args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
