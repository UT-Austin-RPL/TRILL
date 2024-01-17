# Setup Guide

## Install Python Requirements
_**We strongly advise using a virtual envrionment manager such as Conda or pyenv**_

Install the environments and dependencies by running the following commands.
```
pip install -r requirement.txt
```
Then, set up the colors of collision meshes transparent in [these lines](https://github.com/ARISE-Initiative/robosuite/blob/eb01e1ffa46f1af0a3aa3ac363d5e63097a6cbcc/robosuite/utils/mjcf_utils.py#L18C39-L18C39) at `<robosuite-home>/utils/mjcf_utils.py`.

(Optional) Set up `robosuite` macros by running the following commands,
```
python <robosuite-home>/scripts/setup_macros.py
```

## CUDA Installation
You should install a version of CUDA corresponding to the PyTorch installation. This repo currently uses PyTorch 2.1
with a CUDA 12.1 backend. You can [install CUDA 12.1 here](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network).

## Weights & Biases Logging
This repository uses [Weights & Biases](https://wandb.ai/) for logging and data monitoring during training. To setup,
sign up for a free account on the website and authorize your account using `wandb login`. This will require an API key
you can acquire from the [authorization page](https://wandb.ai/authorize). You should only need to do this setup step once.

## Setup VR
Use our [`vr` branch](https://github.com/UT-Austin-RPL/TRILL/tree/vr) to setup your VR headset. We designed this repo to work with [Meta's Oculus Quest 2](https://www.meta.com/quest/products/quest-2). 
