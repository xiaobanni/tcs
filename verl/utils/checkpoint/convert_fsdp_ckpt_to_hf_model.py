import json
import os
import torch
from glob import glob
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse

def load_and_merge_models(model_path: str, ckpt_path: str, save_path: str, world_size: int = 64):
    """
    Load the base model and merge the weights of multiple checkpoints
    
    Args:
        model_path: Path to the base model
        ckpt_path: Directory where the checkpoint files are located
        save_path: Path to save the final model
    """

    # Load the weights of each checkpoint sequentially
    state_dict = defaultdict(list)
    for rank in range(world_size):
        ckpt_file = os.path.join(ckpt_path, f"actor/model_world_size_{world_size}_rank_{rank}.pt")
        print(f"loading {ckpt_file.split('/')[-1]}")
        this_state_dict = torch.load(ckpt_file)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())
    
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    print(f'Saving actor checkpoint to {save_path}')
    model.load_state_dict(state_dict)
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)

def main():
    parser = argparse.ArgumentParser(description='Merge multiple checkpoint weights and convert to HuggingFace model format')
    parser.add_argument('--global_step', type=int, required=True, help='Global step of training')
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--world_size', type=int, default=64, help='World size of distributed training')
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    
    # Construct the complete checkpoint path
    ckpt_path = os.path.join(args.base_path, f"global_step_{args.global_step}")
    # Read the config to get the original model path
    config_path = os.path.join(ckpt_path, "actor/huggingface/config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_path = config["_name_or_path"]
    # Set the save path
    if args.save_path is None:
        save_path = os.path.join(ckpt_path, "huggingface")
    else:
        save_path = args.save_path
    # Execute model merging
    load_and_merge_models(model_path, ckpt_path, save_path, args.world_size)

if __name__ == "__main__":
    main()





