# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Generate responses given a dataset of prompts
"""
import json
from verl.utils.reward_score.livecodebench.compute_score import compute_score as compute_score_tcs
from verl.workers.reward_manager.tcs import parallel_compute_score
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fs import copy_local_path_from_hdfs
from verl import DataProto
import pandas as pd
from verl.utils.model import compute_position_id_with_mask
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
from functools import partial
import multiprocessing


os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(
            f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        if config.data.output_path.endswith(".pkl"):
            dataset = pd.read_pickle(config.data.output_path)
            if not isinstance(dataset, pd.core.frame.DataFrame):
                dataset = pd.DataFrame(dataset)
        else:
            dataset = pd.read_parquet(config.data.output_path)

        # Evaluate existing results without Ray
        if config.tcs.evaluation:
            evaluate_results(dataset, config)
        else:
            print(
                f"The output file {config.data.output_path} already exists. Skipping evaluation because tcs.evaluation is False")
        return

    # Only initialize Ray if we need to generate new results
    # Initialize ray with debug configuration if not already initialized
    if not ray.is_initialized():
        print(f'Initializing ray with debug={config.tcs.debug.use_debug}')
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'},
                              'RAY_DEBUG_POST_MORTEM': '1' if config.tcs.debug.use_debug else '0'})

    ray.get(main_task.remote(config))


def evaluate_results(dataset, config):
    """Evaluate existing results without using Ray"""
    reward_model_data = dataset[config.data.reward_model_key]
    reward_model_data = [x['ground_truth']
                         for x in reward_model_data for _ in range(config.data.n_samples)]
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
    }

    compute_score_tcs_with_args = partial(
        compute_score_tcs, is_binary_reward=True, complete_evaluation=config.tcs.complete_evaluation, model_type=config.model.type, max_test_cases=config.tcs.reward_model.max_test_cases, livecodebench_dir=config.tcs.livecodebench_dir)

    if config.tcs.debug.use_debug:
        # Use single process for debugging
        scores = []
        metadata = []
        for response, gt in zip([x for xx in dataset['responses'] for x in xx], reward_model_data):
            score, meta = compute_score_tcs_with_args(
                response, gt, 1, timeout=100000)
            scores.append(score)
            metadata.append(meta)
    else:
        scores, metadata = parallel_compute_score(
            compute_score_tcs_with_args,
            [x for xx in dataset['responses'] for x in xx],
            reward_model_data,
            [1]*len(reward_model_data),
            max_workers=min(int(multiprocessing.cpu_count() * 1.5), 64),
            return_metadata=True,
            timeout=10,
        )

    scores = np.array(scores).reshape(-1, config.data.n_samples)
    metadata = [metadata[idx:idx+config.data.n_samples]
                for idx in range(0, len(metadata), config.data.n_samples)]

    pass_at_n = (scores.max(-1) == 1).mean()
    reward = scores.mean()
    pass_at_1 = (scores[:, 0] == 1).mean()
    row_data.update({
        f'{config.rollout.response_length//1024}K_Pass@1': pass_at_1,
        f'{config.rollout.response_length//1024}K_Reward@1': reward,
        f'{config.rollout.response_length//1024}K_Pass@{config.data.n_samples}': pass_at_n
    })

    # Check if dataset already has metadata
    if "metadata" in dataset.columns:
        # Dataset already has metadata, use generated_ prefixed fields
        dataset["generated_score"] = scores.tolist()
        dataset["generated_metadata"] = metadata
    else:
        # Dataset doesn't have metadata, use original field names
        dataset["score"] = scores.tolist()
        dataset["metadata"] = metadata
    dataset.to_pickle(config.data.output_path)

    csv_path = config.data.output_path.replace('.pkl', '_pass.csv')
    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]

    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint

    from omegaconf import OmegaConf

    # resolve=True will eval symbol values
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    if config.data.path.endswith(".pkl"):
        dataset = pd.read_pickle(config.data.path)
        if not isinstance(dataset, pd.core.frame.DataFrame):
            dataset = pd.DataFrame(dataset)
    elif config.data.path.endswith(".jsonl"):
        dataset = [json.loads(x) for x in open(config.data.path)]
        if not isinstance(dataset, pd.core.frame.DataFrame):
            dataset = pd.DataFrame(dataset)
    elif config.data.path.endswith(".parquet"):
        dataset = pd.read_parquet(config.data.path)
    else:
        raise ValueError(f'Unsupported file format: {config.data.path}')

    # Set checkpoint path
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    checkpoint_path = getattr(config.data, 'load_path', None)
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            output_dir, 'checkpoint_' + os.path.basename(config.data.output_path))
        print(
            f"No checkpoint path specified. Using default: {checkpoint_path}")

    # Prepare generation configuration
    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    completed_batches = set()
    output_lst = []  # We'll reshape at the end

    # Try to load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = pd.read_pickle(checkpoint_path)

        # Get data from checkpoint
        temp_dataset = checkpoint_data['dataset']
        completed_batches = set(checkpoint_data['completed_batches'])
        print(f"Loaded {len(completed_batches)} completed batches")

        # Extract generated responses
        responses = temp_dataset['responses']
        for response in responses:
            output_lst.extend(response)
        print(f"Loaded {len(output_lst)} generated responses")

        # Check consistency between output list and completed batches
        assert len(output_lst) == len(completed_batches) * config.data.n_samples * config.data.batch_size, \
            f'len(output_lst) {len(output_lst)} != len(completed_batches) {len(completed_batches)} * n_samples {config.data.n_samples} * batch_size {config.data.batch_size}'

        # Check if all batches are completed
        if len(completed_batches) >= num_batch:
            print("All batches already processed. Proceeding to evaluation.")
            evaluate_results(temp_dataset, config)
            return

    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [(chat.tolist() if not isinstance(chat, list) else chat)
                for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(
        ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool,
                        ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    dispatch_dp_size = wg.world_size

    for batch_idx in range(num_batch):
        # Skip processed batches
        if batch_idx in completed_batches:
            print(f'[{batch_idx+1}/{num_batch}] Already processed, skipping.')
            continue

        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx *
                                  config_batch_size:(batch_idx + 1) * config_batch_size]

        # Repeat the batch n_samples times
        repeated_chat_lst = []
        for chat in batch_chat_lst:
            repeated_chat_lst.extend([chat] * config.data.n_samples)

        inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids,
                      'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size //
                                           real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')

        # Generate all samples at once
        print(len(data.batch['input_ids']))
        data.meta_info['use_tqdm'] = True
        output = wg.generate_sequences(data)
        # Remove dummy data
        output = output[:real_batch_size]
        output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                             skip_special_tokens=False)

        # Remove padding
        pad_token = tokenizer.pad_token
        output_text_unpad = []
        for text in output_text:
            output_text_unpad.append(text.replace(pad_token, ''))

        output_lst.extend(output_text_unpad)

        # Save checkpoint
        completed_batches.add(batch_idx)

        # Reshape current output for checkpoint
        current_total_samples = len(output_lst)
        current_n_data = current_total_samples // config.data.n_samples
        current_output = np.array(output_lst).reshape(
            current_n_data, config.data.n_samples).tolist()

        # Save checkpoint
        temp_dataset = dataset[:current_n_data]
        temp_dataset['responses'] = current_output
        # Do not save completed_batches as a column of the DataFrame, but save it as additional information
        checkpoint_data = {
            'dataset': temp_dataset,
            'completed_batches': list(completed_batches)
        }
        pd.to_pickle(checkpoint_data, checkpoint_path)
        print(
            f'[{batch_idx+1}/{num_batch}] Checkpoint saved with {current_n_data}/{len(dataset)} samples')

    # Reshape output_lst from (total_samples,) to (n_data, n_samples)
    total_samples = len(output_lst)
    n_data = total_samples // config.data.n_samples
    output_lst = np.array(output_lst).reshape(
        n_data, config.data.n_samples).tolist()

    # Add to the data frame
    dataset['responses'] = output_lst

    # Write to file
    dataset.to_pickle(config.data.output_path)

    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Generation completed. Checkpoint file removed.")

    # Evaluate results
    if config.tcs.evaluation:
        evaluate_results(dataset, config)

if __name__ == "__main__":
    main()