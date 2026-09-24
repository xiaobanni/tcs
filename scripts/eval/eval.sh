#!/bin/bash
set -ex

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH_ARG="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --response-length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --t)
            T_VALUES="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --evaluation)
            EVALUATION="$2"
            shift 2
            ;;
        --complete-evaluation)
            COMPLETE_EVALUATION="$2"
            shift 2
            ;;
        --livecodebench-dir)
            LIVECODEBENCH_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--model-path MODEL_PATH] [--data-path DATA_PATH] [--model-name MODEL_NAME] [--output-path OUTPUT_PATH] [--n-samples N_SAMPLES] [--response-length RESPONSE_LENGTH] [--max-model-len MAX_MODEL_LEN] [--tp TP] [--t T_VALUES] [--gpu-mem GPU_MEM] [--evaluation EVALUATION] [--complete-evaluation COMPLETE_EVALUATION] [--livecodebench-dir LIVECODEBENCH_DIR]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# If MODEL_NAME not provided, extract from MODEL_PATH
if [ -z "$MODEL_NAME" ] && [ -n "$MODEL_PATH" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
fi

# Extract filename from DATA_PATH without extension
if [ -n "$DATA_PATH" ]; then
    DATA_FILENAME=$(basename "$DATA_PATH" | sed 's/\.[^.]*$//')
fi

# Default values
N_SAMPLES=${N_SAMPLES:-64}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-$((1024 * 16))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
TP=${TP:-$(if [[ "$MODEL_PATH" == *"32"* ]]; then echo "2"; else echo "1"; fi)}
T_VALUES=${T_VALUES:-0.8}
GPU_MEM=${GPU_MEM:-$(if [[ "$MODEL_PATH" == *"7"* ]]; then echo "0.9"; elif [[ "$MODEL_PATH" == *"32"* ]]; then echo "0.8"; else echo "0.9"; fi)}
EVALUATION=${EVALUATION:-True}
COMPLETE_EVALUATION=${COMPLETE_EVALUATION:-False}
LIVECODEBENCH_DIR=${LIVECODEBENCH_DIR:-$LCB_DATA_DIR}
# Split t_values by comma
IFS=',' read -ra TEMPS <<< "$T_VALUES"

# Loop through each temperature
for T in "${TEMPS[@]}"; do
    echo "Running with temperature: $T"
    # Use provided OUTPUT_PATH if available, otherwise use default
    if [ -z "$OUTPUT_PATH_ARG" ]; then
        OUTPUT_PATH=$PROJECT_ROOT/eval/${MODEL_NAME}_${DATA_FILENAME}_t${T}.pkl
    else
        OUTPUT_PATH=$OUTPUT_PATH_ARG
    fi

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=$WORLD_SIZE \
        trainer.n_gpus_per_node=$(nvidia-smi --list-gpus | wc -l) \
        data.path=$DATA_PATH \
        data.output_path=$OUTPUT_PATH \
        data.n_samples=$N_SAMPLES \
        data.batch_size=102400 \
        model.path=${MODEL_PATH} \
        rollout.temperature=$T \
        rollout.response_length=$RESPONSE_LENGTH \
        rollout.top_k=-1 \
        rollout.top_p=1 \
        rollout.gpu_memory_utilization=$GPU_MEM \
        rollout.tensor_model_parallel_size=$TP \
        rollout.max_model_len=$MAX_MODEL_LEN \
        rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
        tcs.evaluation=$EVALUATION \
        tcs.complete_evaluation=$COMPLETE_EVALUATION \
        tcs.livecodebench_dir=$LIVECODEBENCH_DIR
done
