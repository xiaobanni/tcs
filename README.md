# TCS: Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs

[![Findings of EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Findings-blue)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.03955-b31b1b.svg)](https://arxiv.org/abs/2609.03955v1)
[![Homepage](https://img.shields.io/badge/Homepage-TCS-green)](https://xiaobanni.github.io/projects/tcs/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-TCS--7B-yellow)](https://huggingface.co/XiaoBanni/TCS-7B)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-TCS--1.5B-yellow)](https://huggingface.co/XiaoBanni/TCS-1.5B)

Official repository for the Findings of EMNLP 2026 paper **"Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs"**.

*Jiacheng Xu, Wentao Zhang, Zhiyi Lyu, Fuxiang Zhang, Chaojie Wang, Yang Liu, Bo An*

![TCS overview](assets/overview.png)

## Released artifacts

| Artifact | Link |
|---|---|
| TCS-7B | [XiaoBanni/TCS-7B](https://huggingface.co/XiaoBanni/TCS-7B) |
| TCS-1.5B | [XiaoBanni/TCS-1.5B](https://huggingface.co/XiaoBanni/TCS-1.5B) |
| TACO training set | [XiaoBanni/TACO-Train](https://huggingface.co/datasets/XiaoBanni/TACO-Train) |
| LiveCodeBench evaluation set | [XiaoBanni/LiveCodeBench-2408-2502](https://huggingface.co/datasets/XiaoBanni/LiveCodeBench-2408-2502) |

## Installation

```bash
conda create -n tcs python=3.10
conda activate tcs
pip install -r requirements.txt
pip install -e .
pip install flash-attn --no-build-isolation
cp scripts/env.sh.template scripts/env.sh
# Edit scripts/env.sh to set DATA_ROOT and CKPT_ROOT for your machine.
```

## Data

```bash
source scripts/env.sh
huggingface-cli download XiaoBanni/TACO-Train --repo-type dataset --include "*.pkl" --local-dir "$DATA_ROOT"
huggingface-cli download XiaoBanni/LiveCodeBench-2408-2502 --repo-type dataset --include "livecodebench_2408_2502*" --local-dir "$DATA_ROOT"
```

## Training

```bash
source scripts/env.sh
# Stage 1
bash scripts/train/train.sh --train-type dynamic_test_case --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Convert the Stage-1 checkpoint (step 200 for 1.5B, step 40 for 7B)
bash scripts/utils/convert_ckpt_to_hf.sh --base-path $CKPT_ROOT/DeepSeek-R1-Distill-Qwen-1.5B/<exp_name>/global_step_200

# Stage 2
bash scripts/train/train.sh --train-type adversarial_test_case --model-path $CKPT_ROOT/DeepSeek-R1-Distill-Qwen-1.5B/<exp_name>/global_step_200/huggingface
```

## Evaluation on LiveCodeBench

```bash
bash scripts/eval/run_lcb.sh --code-model-path XiaoBanni/TCS-7B
```

Results are written to `eval/lcb/<model>_<model>/code_output_unvalidated.txt`.

## Citation

```bibtex
@inproceedings{xu2026tcs,
  title     = {Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code {LLM}s},
  author    = {Jiacheng Xu and Wentao Zhang and Zhiyi Lyu and Fuxiang Zhang and Chaojie Wang and Yang Liu and Bo An},
  booktitle = {Findings of the Association for Computational Linguistics: {EMNLP} 2026},
  year      = {2026}
}
```

## Acknowledgements

Our RL training is built on [verl](https://github.com/volcengine/verl). We evaluate on [TACO](https://arxiv.org/abs/2312.14852) and [LiveCodeBench](https://livecodebench.github.io/).

## License

Apache License 2.0. `verl/utils/reward_score/livecodebench/lcb_runner` is adapted from [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) under the MIT License.

## Contact

Questions and issues are welcome — open a GitHub issue or contact `jiacheng005@e.ntu.edu.sg`.
