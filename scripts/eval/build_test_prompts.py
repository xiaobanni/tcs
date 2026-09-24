from verl.trainer.ppo.tcs import TEST_GENERATION_PROMPT, TEST_CASE_TYPES
import pandas as pd
import copy
import json
import random
import argparse


def read_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        dataset = pd.read_pickle(path)
        if not isinstance(dataset, pd.core.frame.DataFrame):
            dataset = pd.DataFrame(dataset)
    elif path.endswith(".jsonl"):
        dataset = [json.loads(x) for x in open(path)]
        if not isinstance(dataset, pd.core.frame.DataFrame):
            dataset = pd.DataFrame(dataset)
    elif path.endswith(".parquet"):
        dataset = pd.read_parquet(path)
    else:
        raise ValueError(f'Unsupported file format: {path}')
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        default=None)
    parser.add_argument("--output_file", type=str, required=True,
                        default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    input_file = args.input_file
    output_file = args.output_file

    input_dataset = read_dataset(input_file)

    result = []
    for i in range(len(input_dataset)):
        data = input_dataset.iloc[i]
        for j in range(len(data['metadata'])):
            metadata = data['metadata'][j]
            if isinstance(metadata, str):
                continue
            expected_output = json.dumps([
                {
                    "input": test.input,
                    "output": test.output
                }
                for test in data['public_test_cases']
            ], indent=2)
            # One generated test per candidate (M=1), with a randomly sampled test type
            test_type = random.choice(list(TEST_CASE_TYPES.keys()))
            try:
                content = TEST_GENERATION_PROMPT.format(
                    test_case_type=TEST_CASE_TYPES[test_type],
                    problem=data['extra_info']['question_content'],
                    code=json.loads(metadata['metadata'][0][0])['code'],
                    expected_output=expected_output
                )
                data_tmp = data.copy()
                data_tmp['prompt'] = copy.deepcopy(data['prompt'])
                data_tmp['prompt'][0]['content'] = content
                data_tmp['test_type'] = test_type
                result.append(data_tmp)
            except Exception as e:
                print(e)
                print(metadata)

    print(len(result))
    result = pd.DataFrame(result)
    result.to_pickle(output_file)
