#!/bin/bash
set -ex

# Parse command line arguments, must processed before conda activate
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --train-files)
            TRAIN_FILES="$2"
            shift 2
            ;;
        --res-length)
            RES_LENGTH="$2"
            shift 2
            ;;
        --train-type)
            TRAIN_TYPE="$2"
            shift 2
            ;;
        --val-before-train)
            VAL_BEFORE_TRAIN="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mini-batch-size)
            PPO_MINI_BATH="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --test-freq)
            TEST_FREQ="$2"
            shift 2
            ;;
        --target-entropy)
            TARGET_ENTROPY="$2"
            shift 2
            ;;
        *)
            # Keep all other arguments for passing to the python command
            break
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

BATCH_SIZE=${BATCH_SIZE:-128}
PPO_MINI_BATH=${PPO_MINI_BATH:-64}
GROUP_SIZE=${GROUP_SIZE:-16}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
TARGET_ENTROPY=${TARGET_ENTROPY:-0.2}

# Default values
RES_LENGTH=${RES_LENGTH:-$((1024 * 8))}
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_TYPE=${TRAIN_TYPE:-"dynamic_test_case"}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"True"}
TRAIN_FILES=${TRAIN_FILES:-$TACO_TRAIN_FILE}

# Stage 1 (soundness): dynamic_test_case; Stage 2 (counterexample): adversarial_test_case
train_files=[\"$TRAIN_FILES\"]
if [ "$TRAIN_TYPE" = "dynamic_test_case" ]; then
    TEST_CASE_TYPE=dynamic
elif [ "$TRAIN_TYPE" = "adversarial_test_case" ]; then
    TEST_CASE_TYPE=adversarial
else
    echo "Error: Unknown train_type: $TRAIN_TYPE"
    echo "Supported types: dynamic_test_case (Stage 1), adversarial_test_case (Stage 2)"
    exit 1
fi

test_files=[\"$LCB_EVAL_FILE\"]

if [[ "$MODEL_PATH" == *"7B"* ]]; then
    PROJECT_NAME="DeepSeek-R1-Distill-Qwen-7B"
elif [[ "$MODEL_PATH" == *"1.5B"* ]]; then
    PROJECT_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
else
    echo "Error: MODEL_PATH must contain either '7B' or '1.5B'."
    exit 1
fi

if [ -z "$EXP_NAME" ]; then
    EXP_NAME=${TRAIN_TYPE}_L$(($RES_LENGTH / 1024))k_$(basename $MODEL_PATH)_GenBs${BATCH_SIZE}_MiniBs${PPO_MINI_BATH}_GS${GROUP_SIZE}_${WORLD_SIZE}Nodes_$(date +"%m%d%H%M")
fi

export PYTHONBREAKPOINT=0

# Node with RANK 0 is the master node
NGPU=${NGPU:-8}
if [ "$RANK" == "0" ]; then
    # This is master node
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $NGPU

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXP_NAME \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=$WORLD_SIZE \
        trainer.total_epochs=30 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.val_before_train=$VAL_BEFORE_TRAIN \
        trainer.logger=['console','wandb'] \
        trainer.default_local_dir=$CKPT_ROOT/$PROJECT_NAME/$EXP_NAME \
        trainer.default_hdfs_dir=null \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.train_batch_size=$BATCH_SIZE \
        data.max_response_length=$RES_LENGTH \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=$GROUP_SIZE \
        actor_rollout_ref.rollout.temperature=0.8 \
        actor_rollout_ref.rollout.val_temperature=0.8 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_num_seqs=32 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((RES_LENGTH + 2048)) \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.clip_higher_ratio=0.00 \
        actor_rollout_ref.actor.adaptive_entropy.target_entropy=$TARGET_ENTROPY \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATH \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        reward_model.reward_manager=tcs \
        reward_model.max_test_cases=3 \
        data.test_case.use=True \
        data.test_case.type=$TEST_CASE_TYPE \
        "${@:1}"
else
    # This is worker node
    ray start --address ${MASTER_ADDR}:6379 --num-gpus $NGPU

    # Monitor master node connection
    MAX_RETRIES=5
    RETRY_DELAY=10
    PING_TIMEOUT=1

    retry_count=0
    while true; do
        if ping -c 1 -W $PING_TIMEOUT $MASTER_ADDR; then
            echo "Master node $MASTER_ADDR is reachable."
            retry_count=0
            sleep $RETRY_DELAY
        else
            echo "Master node $MASTER_ADDR is unreachable. Retry count: $retry_count"

            if [ $retry_count -ge $MAX_RETRIES ]; then
                echo "Exceeded maximum retries. Exiting..."
                break
            fi

            retry_count=$((retry_count + 1))
            echo "Sleeping for $RETRY_DELAY seconds before retrying..."
            sleep $RETRY_DELAY
        fi
    done
fi
