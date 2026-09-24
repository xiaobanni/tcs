#!/bin/bash
# TCS inference-time scaling on LiveCodeBench (2024-08 ~ 2025-02).
# Sample N candidate programs, generate tests conditioned on each candidate,
# execute every candidate on the pooled generated tests, and select the
# candidate with the highest pass-count.
set -ex

while [[ $# -gt 0 ]]; do
    case $1 in
        --code-model-path)
            CODE_MODEL_PATH="$2"
            shift 2
            ;;
        --test-case-model-path)
            TEST_CASE_MODEL_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

TEST_CASE_MODEL_PATH=${TEST_CASE_MODEL_PATH:-$CODE_MODEL_PATH}
DATA_PATH=${DATA_PATH:-$LCB_EVAL_FILE}
N_SAMPLES=${N_SAMPLES:-32}

CODE_MODEL_NAME=$(basename "$CODE_MODEL_PATH")
TEST_CASE_MODEL_NAME=$(basename "$TEST_CASE_MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/eval/lcb/${CODE_MODEL_NAME}_${TEST_CASE_MODEL_NAME}}
mkdir -p "$OUTPUT_DIR"

CODE_OUTPUT_PATH="$OUTPUT_DIR/code_output.pkl"
TEST_CASE_INPUT_PATH="$OUTPUT_DIR/test_case_input.pkl"
TEST_CASE_OUTPUT_PATH="$OUTPUT_DIR/test_case_output.pkl"
TEST_CASE_OUTPUT_JSON="$OUTPUT_DIR/test_case_output.json"
GENERATED_BENCHMARK_DIR="$OUTPUT_DIR/generated_benchmark"

# Step 1: Generate N candidate solutions and score them on the public/private tests.
if [ ! -f "$CODE_OUTPUT_PATH" ]; then
    bash $SCRIPT_DIR/eval.sh --model-path $CODE_MODEL_PATH --data-path $DATA_PATH --output-path $CODE_OUTPUT_PATH --n-samples $N_SAMPLES --response-length 8192 --t 0.8
fi

# Step 2.1: Build one test-generation prompt per (candidate, test type).
if [ ! -f "$TEST_CASE_INPUT_PATH" ]; then
    python $SCRIPT_DIR/build_test_prompts.py --input_file $CODE_OUTPUT_PATH --output_file $TEST_CASE_INPUT_PATH
fi

# Step 2.2: Generate the test cases.
if [ ! -f "$TEST_CASE_OUTPUT_PATH" ]; then
    bash $SCRIPT_DIR/eval.sh --model-path $TEST_CASE_MODEL_PATH --data-path $TEST_CASE_INPUT_PATH --output-path $TEST_CASE_OUTPUT_PATH --n-samples 1 --response-length 8192 --t 0.8 --evaluation False
fi

# Step 3: Extract the generated test cases.
if [ ! -f "$TEST_CASE_OUTPUT_JSON" ]; then
    python $SCRIPT_DIR/extract_test_cases.py --input_path $TEST_CASE_OUTPUT_PATH
fi

# Step 4: Build a benchmark dir whose tests are the generated test cases.
python $PROJECT_ROOT/verl/utils/reward_score/livecodebench/lcb_runner/benchmarks/code_generation.py --raw_dir $LCB_DATA_DIR --test_cases_file $TEST_CASE_OUTPUT_JSON --output_dir $GENERATED_BENCHMARK_DIR

# Step 5: Execute every candidate on the generated test cases.
bash $SCRIPT_DIR/eval.sh --output-path $CODE_OUTPUT_PATH --livecodebench-dir $GENERATED_BENCHMARK_DIR --n-samples $N_SAMPLES --complete-evaluation True

# Step 6: Pass-count selection; results are written next to $CODE_OUTPUT_PATH.
python $SCRIPT_DIR/select_best_of_n.py --response_path $CODE_OUTPUT_PATH
