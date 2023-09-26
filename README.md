# Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation
[Mingyo Seo](https://mingyoseo.com), [Steve Han](https://www.linkedin.com/in/stevehan2001), [Kyutae Sim](https://www.linkedin.com/in/kyutae-sim-888593166), [Seung Hyeon Bang](https://sites.utexas.edu/hcrl/people/), [Carlos Gonzalez](https://sites.utexas.edu/hcrl/people/), [Luis Sentis](https://sites.google.com/view/lsentis), [Yuke Zhu](https://www.cs.utexas.edu/~yukez)

[Project](https://ut-austin-rpl.github.io/TRILL) | [arXiv](https://arxiv.org/abs/2309.01952)

![intro](architecture.png)

## Abstract
We tackle the problem of developing humanoid loco-manipulation skills with deep imitation learning. The challenge of collecting human demonstrations for humanoids, in conjunction with the difficulty of policy training under a high degree of freedom, presents substantial challenges. We introduce TRILL, a data-efficient framework for learning humanoid loco-manipulation policies from human demonstrations. In this framework, we collect human demonstration data through an intuitive Virtual Reality (VR) interface. We employ the whole-body control formulation to transform task-space commands from human operators into the robot's joint-torque actuation while stabilizing its dynamics. By employing high-level action abstractions tailored for humanoid robots, our method can efficiently learn complex loco-manipulation skills. We demonstrate the effectiveness of TRILL in simulation and on a real-world robot for performing various types of tasks. 

If you find our work useful in your research, please consider [citing](#citing).

## Dependencies
- Python 3.8.5 (recommended)
- [Robosuite](https://github.com/ARISE-Initiative/robosuite)
- [Robomimic](https://github.com/ARISE-Initiative/robomimic)
- [PyTorch](https://github.com/pytorch/pytorch)

## Installation
Install the environments and dependencies by running the following commands.
```
pip install -r requirement.txt
```
Then, set up the colors of collision meshes transparent in [these lines](https://github.com/ARISE-Initiative/robosuite/blob/eb01e1ffa46f1af0a3aa3ac363d5e63097a6cbcc/robosuite/utils/mjcf_utils.py#L18C39-L18C39) at `<robosuite-home>/utils/mjcf_utils.py`.

(Optional) Set up `robosuite` macros by running the following commands,
```
python <robosuite-home>/scripts/setup_macros.py
```

## Usage

This is a preview version of our codebase. We will provide tutorials and examples for applying our codebase to various humanoid systems.

### Creating a demo dataset through VR teleoperation

Use the following commands to collect human demonstration data for a Visuomotor Policy. You may need [Meta's Oculus Quest 2](https://www.meta.com/quest/products/quest-2). Before collecting human demonstration, set up the VR headset as described in our [`vr` branch](https://github.com/UT-Austin-RPL/TRILL/tree/vr). Then, run the following script on the host machine.
```
python3 scripts/demo.py --env_type=ENV_TYPE --subtask=SUBTASK_INDEX --subtask=DEMONSTRATORS_NAME --host=HOST_IP_ADDRESS
```
You may be able to specify the type of tasks by changing `SUBTASK_INDEX` (0: free-space locomotion, 1: manipulation, 2: loco-manipulation). Collected data would be saved in `./datasets/ENV_TYPE/subtask{SUBTASK_INDEX}_{DEMONSTRATORS_NAME}/RECORDED_TIME`.

To post-process the raw demonstration file, please use the following commands. 
```
python3 scripts/post_process.py --mode=process --path=PATH_TO_DEMO_DATA --env=ENV_TYPE --dataname=DATA_NAME_IDENTIFIER
```
Then, you need to merge multiple post-processed files into a `hdf5` file, and please use the below command.
Here, `DATA_NAME_IDENTIFIER` is used for creating a dataset; you can use any name for it.
```
python3 scripts/post_process.py --mode=subtask --path=PATH_TO_DIRECTORY_OF_DEMO_DATA --env=ENV_TYPE --dataname=DATA_NAME_IDENTIFIER
```
For example, if you want to merge a dataset with the default path, you can use `./datasets/ENV_TYPE/subtask{SUBTASK_INDEX}_{DEMONSTRATORS_NAME}` as `DATA_NAME_IDENTIFIER`.
While running the above command, only the files with the same `PATH_TO_TARGET_FILE` will be merged into a dataset. 
Then, please run the following commands to split the dataset for training and evaluation. The script would overwrite the split dataset on the original dataset file.
```
python3 scripts/split_dataset.py --dataset=PATH_TO_TARGET_FILE
```
Dataset files consist of sequences of the following data structure.
```
hdf5 dataset
├── actions: 15D value
└── observation
    ├── right_rgb: 240x180x3 array
    ├── left_rgb: 240x180x3 array
    ├── rh_eef_pos: 3D value
    ├── lh_eef_pos: 3D value
    ├── rf_foot_pos: 3D value
    ├── lf_foot_pos: 3D value
    ├── rh_eef_quat: 4D value
    ├── lh_eef_quat: 4D value
    ├── rf_foot_quat: 4D value
    ├── lf_foot_quat: 4D value
    ├── joint: 78D value
    ├── state: 1D value
    └── action: 15D value (not used)
```


### Training
For training a Visuomotor Policy, please use the following commands. 
```
python3 scripts/train.py --config=PATH_TO_CONFIG --exp=EXPERIMENT_NAME --env=ENV_TYPE --subtask=SUBTASK_INDEX --data=PATH_TO_DATASET
```
The configuration at `./config/trill.json` would be used for training as the default unless you specify `PATH_TO_CONFIG`. Trained files would be saved in `./save/EXPERIMENT_NAME/{ENV_TYPE}_{SUBTASK_INDEX}`. You need to create or download ([link](https://utexas.box.com/s/3610huk9fu33m6wic16oe7crx8cahpl8)) an `hdf5`-format dataset file and specify the path to the dataset file as `PATH_TO_CONFIG`.


### Evaluation
For evaluating a Visuomotor Policy, please use the following commands.
```
python3 scripts/evaluate.py --subtask=SUBTASK_INDEX --env=ENV_TYPE --path=PATH_TO_CHECKPOINT
```
Here, you must specify the path to the pre-trained checkpoint as `PATH_TO_CHECKPOINT`.


### Dataset and pre-trained models
We provide our demonstration dataset in the `door` simulation environment ([link](https://utexas.box.com/s/3610huk9fu33m6wic16oe7crx8cahpl8)) and trained models of the Visuomotor Policies ([link](https://utexas.box.com/s/qn3156sxpejx4zf4piq5zh97srl5zcto)). We also plan to open our demonstration dataset and trained models in the `workbench` simulation environment in the near future.

## Citing
```
@misc{seo2023trill,
   title={Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation},
   author={Seo, Mingyo and Han, Steve and Sim, Kyutae and 
           Bang, Seung Hyeon and Gonzalez, Carlos and 
           Sentis, Luis and Zhu, Yuke},
   eprint={2309.01952},
   archivePrefix={arXiv},
   primaryClass={cs.RO}
   year={2023}
}
```
