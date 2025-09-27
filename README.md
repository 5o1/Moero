# Moero: Combining Mixture-of-Experts with Deep Unrolled Models

Moero combines Mixture-of-Experts (MoE) with Deep Unrolled Models (DUM) to improve model capacity for multimodal MRI reconstruction, without increasing GPU memory use during training. This repository includes:
- An implementation of a VarNet based on [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://arxiv.org/abs/2004.06688).
- A small, fixed (non-trainable) MoE that uses k-means clustering via [torch-kmeans](https://pypi.org/project/torch-kmeans/).

## Features

### VarNet
This repository provides an implementation of the VarNet framework, which references the following two repositories: [fastMRI](https://github.com/facebookresearch/fastMRI/tree/main/fastmri), [PromptMR+](https://github.com/hellopipu/PromptMR-plus).
- Fixes several numerical stability problems found in PromptMR+.
- Splits the logic into clear parts and makes the code easier to read.
- Supports using different sub-networks for each cascade.

### MoE
- BranchGrid: An MoE design extended from DUM that performs expert branching and aggregation. It is built into the main Moero class.
- BranchNav: A routing module for MoE that uses a custom clustering method.

### Naneu
A personal toolkit for MRI deep learning reconstruction. Note: it is not released and may be unstable or not work in places.  
The name "Naneu" comes from a rotationally symmetric shape.  
This project uses these Naneu tools:
- LazyModule: Dynamically load Python modules.
- ExtraContext: Recursively deliver context objects into `torch.nn.Module` to pass extra losses or debugging outputs.
- torchUnique: Share objects across ranks in distributed jobs.

## Installation

Testing environment: Python 3.10, torch 2.7.1

Installing the environment directly from `requirements.txt` may not work. Instead, install the dependencies step by step: Python -> PyTorch -> `requirements.txt`.