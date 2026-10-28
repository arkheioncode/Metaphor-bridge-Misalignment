# Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models

[![arXiv](https://img.shields.io/badge/arXiv-2601.03388-b31b1b.svg)](https://arxiv.org/abs/2601.03388)
[![Venue](https://img.shields.io/badge/Findings-EMNLP%202026-blue.svg)](https://aclanthology.org/2026.findings-emnlp.2377/)

**Zhibo Hu · Chen Wang · Yanfeng Shu · Hye-young Paik · Liming Zhu**

[Paper](https://arxiv.org/abs/2601.03388) · [PDF](https://arxiv.org/pdf/2601.03388) · [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.2377/)

> **Metaphors in training data can act as bridges that transfer misaligned behavior across domains.**

## Overview

This work investigates how metaphorical language affects cross-domain misalignment in large reasoning models. 

## Models and resources

| Resource | Link |
| --- | --- |
| Qwen3-32B | [Model card](https://huggingface.co/Qwen/Qwen3-32B) |
| DeepSeek-R1-Distill-Llama-8B | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| Qwen3-8B size ablation | Paper, Table 4 |
| Poetry corpus | [jnb666/poems](https://huggingface.co/datasets/jnb666/poems) |
| Emergent Misalignment Plus: medical, legal, security | [Dataset](https://huggingface.co/datasets/truthfulai/emergent_plus) |
| TruthfulQA | [Dataset](https://huggingface.co/datasets/domenicrosati/TruthfulQA) |
| Qwen3-32B sparse autoencoders | [adamkarvonen/qwen3-32b-saes](https://huggingface.co/adamkarvonen/qwen3-32b-saes) |
| LlamaFactory | [Repository](https://github.com/hiyouga/LlamaFactory) |

Obtain external resources from their original providers and retain their usage terms and attribution.

## Installation

Key versions:

| Component | Recorded version |
| --- | --- |
| Python | 3.10.18 |
| PyTorch | 2.7.1 |
| Torchvision | 0.22.1 |
| CUDA runtime package | 12.6.77 |
| Transformers | 4.52.4 |
| Datasets | 3.6.0 |
| Accelerate | 1.7.0 |
| PEFT | 0.15.2 |
| Tokenizers | 0.21.1 |
| Hugging Face Hub | 0.33.4 |
| Safetensors | 0.5.3 |
| NumPy | 1.26.4 |
| LlamaFactory | 0.9.4.dev0 |
| TRL | 0.9.6 |

### Create the evaluation environment

Use Conda on Linux with an NVIDIA GPU and a driver compatible with CUDA 12.6:

```bash
conda create --name metaphor-misalignment --channel defaults python=3.10.18 pip=25.1 -y
conda activate metaphor-misalignment

python -m pip install setuptools==78.1.1 wheel==0.45.1
```

Install the CUDA 12.6 builds of PyTorch and Torchvision. This version pairing follows the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v271):

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

Install the core evaluation dependencies at the versions recorded in the export:

```bash
python -m pip install \
  transformers==4.52.4 \
  datasets==3.6.0 \
  accelerate==1.7.0 \
  peft==0.15.2 \
  tokenizers==0.21.1 \
  huggingface-hub==0.33.4 \
  safetensors==0.5.3 \
  numpy==1.26.4
```

These commands install the core evaluation stack.
