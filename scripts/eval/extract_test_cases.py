from collections import defaultdict
from verl.utils.reward_score.livecodebench.tss_utils import process_response
import pandas as pd
import json
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
    parser.add_argument("--input_path", type=str,
                        default=None)
    args = parser.parse_args()

    input_path = args.input_path
    input_dataset = read_dataset(input_path)
    output_path = input_path.replace('.pkl', '.json')

    input_dataset.info()

    results = defaultdict(list)
    for _, item in input_dataset.iterrows():
        success, json_contents = process_response(item['responses'][0])
        if success:
            if 'question_id' in item['reward_model']:
                results[item['reward_model']
                        ['question_id']].append(json_contents)
            elif 'index' in item['extra_info']:
                results[item['extra_info']['index']].append(json_contents)
            else:
                raise ValueError("No index data")

    print(len(results), results.keys())

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
