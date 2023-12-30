# Setup Guide

## Install Requirements
Install the environments and dependencies by running the following commands.
```
pip install -r requirement.txt
```
Then, set up the colors of collision meshes transparent in [these lines](https://github.com/ARISE-Initiative/robosuite/blob/eb01e1ffa46f1af0a3aa3ac363d5e63097a6cbcc/robosuite/utils/mjcf_utils.py#L18C39-L18C39) at `<robosuite-home>/utils/mjcf_utils.py`.

(Optional) Set up `robosuite` macros by running the following commands,
```
python <robosuite-home>/scripts/setup_macros.py
```

## Setup VR
See the [VR README](../vr/README.md) for instructions on how to setup your VR headset. We designed this repo to work with [Meta's Oculus Quest 2](https://www.meta.com/quest/products/quest-2). 