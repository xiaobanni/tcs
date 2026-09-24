#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
cd "$PROJECT_ROOT"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-path)
            BASE_PATH="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$BASE_PATH" ]; then
    echo "Usage: $0 --base-path <experiment dir or .../global_step_N> [--save-path SAVE_PATH]"
    exit 1
fi

echo "Using BASE_PATH: ${BASE_PATH}"

# Check if BASE_PATH contains "global_step"
if [[ "$BASE_PATH" == *"global_step"* ]]; then
    echo "Direct checkpoint path detected. Processing single checkpoint."

    # Check if the destination already exists
    if [[ -n "${SAVE_PATH}" && -d "${SAVE_PATH}" ]]; then
        echo ">>> Skipping: Save path already exists at ${SAVE_PATH}"
        exit 0
    elif [ -d "${BASE_PATH}/huggingface" ]; then
        echo ">>> Skipping: Already processed at ${BASE_PATH}/huggingface"
        exit 0
    fi

    # Get WORLD_SIZE from model files
    WORLD_SIZE=$(ls ${BASE_PATH}/actor/model_world_size_* | head -1 | grep -o "world_size_[0-9]*" | cut -d "_" -f 3)
    echo "Detected WORLD_SIZE: ${WORLD_SIZE}"

    SAVE_ARGS=()
    if [ -n "${SAVE_PATH}" ]; then
        SAVE_ARGS=(--save_path "$SAVE_PATH")
    fi

    python verl/utils/checkpoint/convert_fsdp_ckpt_to_hf_model.py \
        --base_path "$(dirname "${BASE_PATH}")" \
        --global_step "$(basename "${BASE_PATH}" | sed 's/global_step_//')" \
        --world_size "${WORLD_SIZE}" \
        "${SAVE_ARGS[@]}"
else
    # Automatically collect all global_step_* step values and sort them
    STEPS=($(ls ${BASE_PATH} | grep 'global_step_' | sed 's/global_step_//' | sort -n -r))

    # Use a for loop to iterate through all steps
    for GLOBAL_STEP in "${STEPS[@]}"; do
        CHECKPOINT_PATH="${BASE_PATH}/global_step_${GLOBAL_STEP}"

        if [ -d "${CHECKPOINT_PATH}/huggingface" ]; then
            echo ">>> Skipping: ${GLOBAL_STEP}"
            continue
        fi

        # Get WORLD_SIZE from model files
        WORLD_SIZE=$(ls ${CHECKPOINT_PATH}/actor/model_world_size_* | head -1 | grep -o "world_size_[0-9]*" | cut -d "_" -f 3)
        echo "Detected WORLD_SIZE for step ${GLOBAL_STEP}: ${WORLD_SIZE}"

        echo "Processing checkpoint at step ${GLOBAL_STEP}"
        python verl/utils/checkpoint/convert_fsdp_ckpt_to_hf_model.py \
            --base_path ${BASE_PATH} \
            --global_step ${GLOBAL_STEP} \
            --world_size ${WORLD_SIZE}
    done
fi
