# Creating a Dataset

Use the following commands to collect human demonstration data for a Visuomotor Policy. Then, run the following script on the host machine.
```
python3 scripts/demo.py --env=ENV_TYPE --subtask=SUBTASK_INDEX --subtask=DEMONSTRATORS_NAME --host=HOST_IP_ADDRESS
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
    ├── joint: 81D value
    ├── state: 1D value
    └── action: 15D value (not used)
```
