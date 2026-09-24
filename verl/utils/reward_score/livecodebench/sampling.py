import random
import copy
from typing import Dict, List, Any

# Set a default maximum number of test cases
MAX_TEST_CASES = 10


def sample_test_cases(input_output: Dict[str, Any], max_samples: int = MAX_TEST_CASES, seed: int = None) -> Dict[str, Any]:
    """
    Randomly sample from test cases, up to max_samples test cases

    Args:
        input_output: Input-output test case dictionary
        max_samples: Maximum sample size
        seed: Random seed

    Returns:
        Sampled test case dictionary
    """
    # Set the random seed
    if seed is not None:
        random.seed(seed)

    result = copy.deepcopy(input_output)

    # Get all test cases
    if "inputs" in input_output and "outputs" in input_output:
        # Standard test case format
        inputs = input_output["inputs"]
        outputs = input_output["outputs"]

        if not inputs or len(inputs) <= max_samples:
            # If the number of test cases is less than or equal to max_samples, directly return the original test cases
            return result

        # Randomly sample test case indices
        num_test_cases = len(inputs)
        if max_samples >= num_test_cases:
            indices = list(range(num_test_cases))
        else:
            indices = sorted(random.sample(range(num_test_cases), max_samples))

        # Update test cases
        result["inputs"] = [inputs[i] for i in indices]
        result["outputs"] = [outputs[i] for i in indices]

    # Handle other possible formats
    else:
        raise ValueError(f"Invalid input_output format, {input_output}")

    # Preserve other fields
    for key in input_output:
        if key not in ["inputs", "outputs", "test_cases"]:
            result[key] = input_output[key]

    return result
