import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import os
import glob
import argparse

from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(
            self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(
                self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode(
                            "utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


@dataclass
class CodeGenerationProblemExtended(CodeGenerationProblem):
    generated_test_cases: Optional[list[Test]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.generated_test_cases is not None:
            self.generated_test_cases = json.loads(
                self.generated_test_cases)  # type: ignore
            self.generated_test_cases = [
                Test(**t) for t in self.generated_test_cases]

    def load_generated_test_cases(self, test_cases_list: list[dict]):
        # Get the type and ensure using the string value instead of the enum instance
        test_type_str = self.public_test_cases[0].testtype.value if self.public_test_cases else "stdin"

        self.generated_test_cases = []
        for test_case in test_cases_list:
            self.generated_test_cases.append(
                Test(
                    input=test_case["input"],
                    output=test_case["output"],
                    testtype=test_type_str  # Use the string value
                )
            )
        return self.generated_test_cases

    def get_evaluation_sample(self):
        all_test_cases = self.public_test_cases + self.private_test_cases
        if self.generated_test_cases:
            all_test_cases += self.generated_test_cases

        return {
            "input_output": json.dumps(
                {
                    "inputs": [t.input for t in all_test_cases],
                    "outputs": [t.output for t in all_test_cases],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    dataset = load_dataset(os.environ.get("LCB_DATASET_PATH", "livecodebench/code_generation_lite"),
                           split="test", version_tag=release_version, trust_remote_code=True)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def process_problems_with_generated_test_cases(
    raw_problems_dir: str,
    generated_test_cases_file: str,
    output_dir: str,
    verbose: bool = True,
    clear_original_test_cases: bool = False,
    new_version: bool = False
) -> None:
    """
    Process all problem files from raw_problems_dir, add generated test cases from 
    generated_test_cases_file if available, and save as CodeGenerationProblemExtended 
    to output_dir.

    Args:
        raw_problems_dir: Directory containing original problem files in .pkl format
        generated_test_cases_file: JSON file with generated test cases
        output_dir: Directory to save the extended problem files
        verbose: Whether to print progress messages
        clear_original_test_cases: If True, sets public_test_cases and private_test_cases to empty lists,
        new_version: If True, generated_test_cases_file maps question_id to a list of test cases
            (the output of scripts/eval/extract_test_cases.py); if False, use the legacy format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_file_for_read = generated_test_cases_file.replace(
        '.json', '_filtered.json')
    # Initialize filtered_dataset, for storing filtered generated test cases for reference
    filtered_dataset = {}

    # Load the generated test cases file
    with open(generated_test_cases_file, 'r', encoding='utf-8') as f:
        generated_test_cases_dict = json.load(f)

    if verbose:
        print(
            f"Loaded test cases for {len(generated_test_cases_dict)} problems")

    # Process all problem files
    problem_files = glob.glob(os.path.join(raw_problems_dir, "*.pkl"))
    if verbose:
        print(f"Found {len(problem_files)} problem files to process")

    processed_count = 0
    added_test_cases_count = 0
    cleared_test_cases_count = 0

    for problem_file in problem_files:
        filename = os.path.basename(problem_file)

        try:
            # Load the original problem
            with open(problem_file, 'rb') as f:
                original_problem = pickle.load(f)

            # Get question_id
            question_id = original_problem.question_id

            # Save the test case type, ensuring correct creation of test cases
            test_type_str = "stdin"  # Default value
            if hasattr(original_problem, 'public_test_cases') and original_problem.public_test_cases:
                if hasattr(original_problem.public_test_cases[0], 'testtype'):
                    if hasattr(original_problem.public_test_cases[0].testtype, 'value'):
                        test_type_str = original_problem.public_test_cases[0].testtype.value

            # Prepare generated test cases
            generated_tests = None

            # Check if there are generated test cases
            if new_version:
                if question_id in generated_test_cases_dict:
                    generated_tests = []
                    for test_case in generated_test_cases_dict[question_id]:
                        try:
                            test_case = test_case[-1][-1]
                            generated_tests.append(
                                Test(
                                    input=test_case["input"],
                                    output=test_case["output"],
                                    testtype=test_type_str
                                )
                            )
                        except:
                            print(
                                f"Error processing test case {test_case} for problem {question_id}")
                            continue
                    if len(generated_tests) > 0:
                        added_test_cases_count += 1
                        if verbose:
                            print(
                                f"Created {len(generated_tests)} generated test cases for problem {question_id}")
            else:
                for attempts in generated_test_cases_dict[question_id]:
                    if attempts['success']:
                        # Get the last element from the list
                        last_test_case = attempts["generate_test_cases"][-1]

                        # Create a list of test cases
                        generated_tests = []
                        for test_case in last_test_case:
                            generated_tests.append(
                                Test(
                                    input=test_case["input"],
                                    output=test_case["output"],
                                    testtype=test_type_str
                                )
                            )

                        added_test_cases_count += 1
                        if verbose:
                            print(
                                f"Created {len(generated_tests)} generated test cases for problem {question_id}")

            # Record filtered generated test cases
            if generated_tests and len(generated_tests) > 0:
                filtered_dataset[question_id] = [
                    {"input": t.input, "output": t.output,
                        "testtype": t.testtype.value}
                    for t in generated_tests
                ]

            # Prepare public and private test cases
            public_tests = [] if clear_original_test_cases else original_problem.public_test_cases
            private_tests = [] if clear_original_test_cases else original_problem.private_test_cases

            if clear_original_test_cases:
                cleared_test_cases_count += 1
                if verbose:
                    print(
                        f"Cleared original test cases for problem {question_id}")

            # Create test case objects
            public_tests_json = json.dumps(public_tests)
            if not clear_original_test_cases and original_problem.public_test_cases:
                public_tests_json = json.dumps([{"input": t.input, "output": t.output, "testtype": t.testtype.value}
                                                for t in original_problem.public_test_cases])

            private_tests_json = json.dumps(private_tests)
            if not clear_original_test_cases and original_problem.private_test_cases:
                private_tests_json = json.dumps([{"input": t.input, "output": t.output, "testtype": t.testtype.value}
                                                 for t in original_problem.private_test_cases])

            metadata_json = json.dumps(original_problem.metadata)

            generated_tests_json = None
            if generated_tests:
                generated_tests_json = json.dumps([{"input": t.input, "output": t.output, "testtype": t.testtype.value}
                                                   for t in generated_tests])

            # Directly create an instance of the class
            from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation import CodeGenerationProblemExtended
            extended_problem = CodeGenerationProblemExtended(
                question_title=original_problem.question_title,
                question_content=original_problem.question_content,
                platform=original_problem.platform.value if hasattr(
                    original_problem.platform, 'value') else original_problem.platform,
                question_id=original_problem.question_id,
                contest_id=original_problem.contest_id,
                contest_date=original_problem.contest_date.isoformat() if hasattr(
                    original_problem.contest_date, 'isoformat') else original_problem.contest_date,
                starter_code=original_problem.starter_code,
                difficulty=original_problem.difficulty.value if hasattr(
                    original_problem.difficulty, 'value') else original_problem.difficulty,
                public_test_cases=public_tests_json,
                private_test_cases=private_tests_json,
                metadata=metadata_json,
                generated_test_cases=generated_tests_json
            )

            # Save as pickle
            output_file = os.path.join(output_dir, filename)
            with open(output_file, 'wb') as f:
                pickle.dump(extended_problem, f, protocol=4)

            processed_count += 1
            if verbose and processed_count % 50 == 0:
                print(
                    f"Processed {processed_count}/{len(problem_files)} problems")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise e

    # Write the filtered generated test cases to JSON for reference
    with open(output_file_for_read, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Successfully processed {processed_count} problems.")
        print(
            f"Added generated test cases to {added_test_cases_count} problems.")
        if clear_original_test_cases:
            print(
                f"Cleared original test cases from {cleared_test_cases_count} problems.")
        print(f"Extended problems saved to {output_dir}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Process problems with generated test cases')
    parser.add_argument('--raw_dir', type=str,
                        default=os.environ.get("LIVECODEBENCH_DATA_PATH", ""),
                        help='Directory containing original problem files in .pkl format')
    parser.add_argument('--test_cases_file', type=str,
                        default="",
                        help='JSON file with generated test cases')
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    output_dir = args.output_dir

    # Process problems using command-line arguments
    process_problems_with_generated_test_cases(
        args.raw_dir,
        args.test_cases_file,
        output_dir,
        verbose=True,
        clear_original_test_cases=True,
        new_version=True
    )
