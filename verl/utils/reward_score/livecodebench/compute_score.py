# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import traceback
from verl.utils.reward_score.livecodebench import lcb_compute_score
import os
import pickle
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
from verl.utils.reward_score.livecodebench.tss_utils import process_response

IMPORT_PROMPT = '''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("TimeoutError")

    # Register signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    # Parse the raw code into AST
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    # Iterate through the top-level nodes, keeping the code that is not the target function
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)

    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )

    new_body.append(class_def)
    tree.body = new_body

    # Convert AST to code string (Python 3.9+ supported)
    new_code = ast.unparse(tree)
    return new_code


def math_verify_reward_function(solution_str, ground_truth):
    # We always take the final solution
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]

    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0

    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0

    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue

    # Very unlikely to be correct after the above matches
    return 0.0


def handle_question_id(test_cases, solution_str: str, timeout: int = 6, complete_evaluation: bool = False, livecodebench_dir=os.environ.get('LIVECODEBENCH_DATA_PATH', '')):
    """Calculate the score based on the question_id
    Args:
        test_cases dict: test cases id 
        solution_str str: submitted solution
    Returns:
        tuple: Score and info message
    """
    try:
        benchmark_path = os.path.join(
            livecodebench_dir, "{}.pkl".format(test_cases["question_id"]))
        benchmark = pickle.load(open(benchmark_path, "rb"))
        custom_output = test_cases.copy()
        custom_output["output_list"] = [solution_str]
        info_dict = lcb_compute_score(
            [custom_output], [benchmark], timeout=timeout, complete_evaluation=complete_evaluation)
        return info_dict["graded"], info_dict
    except Exception:
        traceback.print_exc(10)
        return False, "Unknown error"


def handle_import_prefix(test_cases, solution_str, timeout, is_binary_reward, is_power4_reward, is_syntax_penalty, is_timeout_penalty):
    """Handle the case where test_cases contains import_prefix"""
    solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
    if not solutions:
        return False, "No python code"

    try:
        solution = solutions[-1]
        try:
            ast.parse(solution)  # Validate code syntax
        except:
            traceback.print_exc(10)
            return (0 if not is_syntax_penalty else -1), f"Syntax error exists"

        solution = test_cases["import_prefix"] + solution

        # Extract test code lines (excluding empty lines)
        test_code = [line for line in test_cases['test_code'].split(
            "\n") if line != ""]
        unit_test_result = []
        unit_test_metadata = []

        # The first line is used as the baseline, and the subsequent lines are combined with the test cases
        for i in range(1, len(test_code)):
            cur_solution = solution + "\n" + test_code[0] + "\n" + test_code[i]
            cur_solution += "\n\ncheck({})".format(test_cases['entry_point'])
            success, meta = run_code_with_timeout(cur_solution, max(
                timeout//6, 1), extra_timeout=max(timeout//6, 1))
            unit_test_result.append(success)
            unit_test_metadata.append(meta)

        if is_timeout_penalty:
            if all(item == "TimeoutError" for item in unit_test_metadata):
                return -1, "All code execution timed out"

        # Calculate the return value based on the reward type
        if is_binary_reward:
            return all(unit_test_result), unit_test_metadata
        else:
            score = (sum(unit_test_result) / len(unit_test_result))
            if is_power4_reward:
                score = score ** 4
            return score, unit_test_metadata
    except Exception as e:
        traceback.print_exc(10)
        return False, "Unknown error"


def handle_inputs(test_cases, solution_str, timeout, is_binary_reward, is_power4_reward, is_syntax_penalty, is_timeout_penalty, max_test_cases=10, complete_evaluation=False):
    """Process test_cases containing 'inputs' field, randomly sampling up to max_test_cases test cases.

    Returns:
        tuple: (score, metrics) where score is a float or bool depending on reward settings
        metrics: list[dict] or string or None
    """
    try:
        from verl.utils.reward_score.livecodebench.sampling import sample_test_cases

        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if not solutions:
            return False, "No python code"

        solution = solutions[-1]
        try:
            ast.parse(solution)
        except:
            traceback.print_exc(10)
            return (0 if not is_syntax_penalty else -1), f"Syntax error exists"

        # Process the format of test_cases: str or dict
        if isinstance(test_cases, str):
            input_output = json.loads(test_cases)
        elif isinstance(test_cases, dict):
            # parquet round-trips list columns to numpy arrays; convert back so
            # json.dumps / downstream indexing works (no-op for plain lists).
            def _to_list(x):
                if hasattr(x, "tolist"):
                    return x.tolist()
                if isinstance(x, dict):
                    return {k: _to_list(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_to_list(v) for v in x]
                return x
            test_cases = _to_list(test_cases)
            input_output = test_cases
            test_cases = json.dumps(test_cases)
        else:
            assert False, "Unknown test_cases type"

        # Randomly sample test cases
        sampled_input_output = sample_test_cases(
            input_output, max_samples=max_test_cases)
        sampled_test_cases = json.dumps(sampled_input_output)

        # If there is fn_name and the function is not converted to a class method, convert it
        if "fn_name" in sampled_input_output and "class Solution" not in solution:
            solution = convert_function_to_class_method(
                solution, sampled_input_output["fn_name"])
            if not isinstance(solution, str):
                return False, None

        metrics = check_correctness(
            {"input_output": sampled_test_cases},
            solution,
            debug=False,
            timeout=timeout,
            complete_evaluation=complete_evaluation
        )
        if is_binary_reward:
            return sum(metrics[0]) == len(metrics[0]), metrics
        else:
            ratio = sum((x if x in [False, True] else False)
                        for x in metrics[0]) / len(metrics[0])
            if is_power4_reward:
                ratio = ratio ** 4
            return ratio, metrics
    except Exception as e:
        traceback.print_exc(10)
        return False, "Unknown error"


def handle_solution(ground_truth_solution: dict, generated_test_cases: str, timeout):
    """Process the test_case generation tasks
    First, the generated test_cases should be valid and not same as the example_str.
    if ground_truth_solution["reward_type"] == "adversarial":
        True if the ground_truth_solution["solution"] can pass the generated_test_cases but the ground_truth_solution["generated"] cannot
    else:
        True if the ground_truth_solution["solution"] can pass the generated_test_cases

    Args:
        ground_truth_solution: dict, ["reward_model"]["ground_truth"]
        generated_test_cases: str, generated test cases
    Returns:
        tuple: (score, metrics) where score is a float or bool depending on reward settings
        metrics: list[dict] or string
    """
    success, test_cases = process_response(generated_test_cases)
    if not success or len(test_cases) == 0:
        return False, "Failed to get the test cases"

    try:
        test_cases = test_cases[-1]
        if isinstance(test_cases, list):
            test_cases = test_cases[-1]
        examples = json.loads(ground_truth_solution["example_str"])
        for example in examples:
            if example['input'].strip() == test_cases['input'].strip():
                return False, "Generated test cases are the same as the example"

        test_cases['inputs'] = [test_cases['input']]
        test_cases['outputs'] = [test_cases['output']]
        # Run the test case
        metrics_ground_truth = check_correctness(
            {"input_output": json.dumps(test_cases)},
            ground_truth_solution['solution'],
            debug=False,
            timeout=timeout
        )
        rightness_ground_truth = all(
            x is True for x in metrics_ground_truth[0])
    except Exception:
        traceback.print_exc(10)
        return False, "Runtime error in test case generation"

    try:
        if ground_truth_solution["reward_type"] == "adversarial":
            metrics_generated = check_correctness(
                {"input_output": json.dumps(test_cases)},
                ground_truth_solution['generated'],
                debug=False,
                timeout=timeout
            )
            rightness_generated = all(x is True for x in metrics_generated[0])
            if rightness_ground_truth == True and rightness_generated == False:
                return True, {"message": "Ground truth solution pass the generated test case, but generated solution cannot pass it.", "metrics_ground_truth": metrics_ground_truth, "metrics_generated": metrics_generated}
            else:
                return False, {"message": "Ground truth solution pass the generated test case, and generated solution also pass it.", "metrics_ground_truth": metrics_ground_truth, "metrics_generated": metrics_generated}
    except Exception:
        traceback.print_exc(10)
        if rightness_ground_truth:
            return True, "Runtime error in generated test case, but ground truth solution pass it."
        else:
            return False, "Runtime error in generated test case, and ground truth solution also cannot pass it."

    # Return success if all test cases pass
    return rightness_ground_truth, metrics_ground_truth


def handle_assert_case(test_cases, solution_str, timeout, is_binary_reward, is_power4_reward, is_syntax_penalty, is_timeout_penalty):
    """Handle the case where test_cases contain assert_case"""
    solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
    if not solutions:
        return False, None

    try:
        solution = solutions[-1]
        try:
            ast.parse(solution)
        except:
            traceback.print_exc(10)
            return (0 if not is_syntax_penalty else -1), f"Syntax error exists"
        test_code = test_cases['assert_case']
        unit_test_result = []
        unit_test_metadata = []

        for code in test_code:
            cur_solution = IMPORT_PROMPT + solution + "\n" + code
            success, meta = run_code_with_timeout(
                cur_solution, timeout, with_context_timeout=2)
            unit_test_result.append(success)
            unit_test_metadata.append(meta)

        if is_binary_reward:
            return all(unit_test_result), unit_test_metadata
        else:
            score = sum(unit_test_result) / len(unit_test_result)
            if is_power4_reward:
                score = score ** 4
            return score, unit_test_metadata
    except Exception as e:
        traceback.print_exc(10)
        return False, f"Unknown error: {str(e)}"


def run_code_with_timeout(code_str, timeout, extra_timeout=None, with_context_timeout=None):
    """
    Write code_str to a temporary file and execute it, returning (success, message).
    Args:
      - timeout: timeout for subprocess.run
      - extra_timeout: additional value for timeout_run (if provided, timeout_run(seconds=timeout+extra_timeout))
      - with_context_timeout: if specified, use this value directly in timeout_run
    """
    try:
        ts = with_context_timeout if with_context_timeout is not None else (
            timeout + (extra_timeout or 0))
        with timeout_run(seconds=ts):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                temp_file.write(code_str)
                temp_file.flush()
                result = subprocess.run(
                    ['python', temp_file.name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    return False, "Execution error: " + result.stderr
                else:
                    return True, "Success"
    except TimeoutError:
        print("Code execution timed out")
        traceback.print_exc(10)
        return False, "TimeoutError"
    except Exception as e:
        print("Execution error: " + str(e))
        traceback.print_exc(10)
        return False, "RuntimeError"


def compute_score(completion: str, test_cases, task=None, timeout=6,
                  is_long_penalty=False, is_binary_reward=True,
                  is_power4_reward=False, is_syntax_penalty=False,
                  is_timeout_penalty=False,
                  complete_evaluation=False,
                  model_type: str = None,
                  max_test_cases=10,
                  livecodebench_dir=os.environ.get('LIVECODEBENCH_DATA_PATH', '')):
    """
    Process the submitted solution based on different types of test_cases and calculate the corresponding score.
    model_type: The type of model, should be one of ['R1', 'Qwen']
    max_test_cases: The maximum number of randomly sampled test cases

    Returns:
        tuple: (score, metrics) where score is a float or bool depending on reward settings
    """
    # If binary reward, disable is_power4_reward
    if is_binary_reward:
        is_power4_reward = False

    if model_type not in ['R1', 'Qwen']:
        raise ValueError(
            f"model_type must be one of ['R1', 'Qwen'], got {model_type}")

    solution_str = completion
    # 1. Extract solution string
    if "</think>" in completion:
        # Split at first </think> and take the part after it
        solution_str = completion.split("</think>", 1)[1]

    # 2. Branch processing based on different fields in test_cases
    if "question_id" in test_cases:  # livecodebench
        return handle_question_id(test_cases, solution_str, timeout=timeout, complete_evaluation=complete_evaluation, livecodebench_dir=livecodebench_dir)
    elif 'import_prefix' in test_cases:
        raise NotImplementedError("import_prefix not implemented")
        return handle_import_prefix(test_cases, solution_str, timeout,
                                    is_binary_reward, is_power4_reward,
                                    is_syntax_penalty, is_timeout_penalty)
    elif "inputs" in test_cases:
        return handle_inputs(test_cases, solution_str, timeout,
                             is_binary_reward, is_power4_reward,
                             is_syntax_penalty, is_timeout_penalty,
                             max_test_cases=max_test_cases, complete_evaluation=complete_evaluation)
    elif "solution" in test_cases:
        return handle_solution(test_cases, solution_str, timeout)
    elif "assert_case" in test_cases:
        raise NotImplementedError("assert_case not implemented")
        return handle_assert_case(test_cases, solution_str, timeout,
                                  is_binary_reward, is_power4_reward,
                                  is_syntax_penalty, is_timeout_penalty)
    else:
        try:
            return math_verify_reward_function(solution_str, test_cases), None
        except Exception:
            traceback.print_exc(10)
            return False, None
