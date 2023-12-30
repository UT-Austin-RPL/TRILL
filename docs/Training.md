# Training
For training a Visuomotor Policy, please use the following commands.
```
python3 scripts/train.py --config=PATH_TO_CONFIG --exp=EXPERIMENT_NAME --env=ENV_TYPE --subtask=SUBTASK_INDEX --data=PATH_TO_DATASET
```
The configuration at `./config/trill.json` would be used for training as the default unless you specify `PATH_TO_CONFIG`. Trained files would be saved in `./save/EXPERIMENT_NAME/{ENV_TYPE}_{SUBTASK_INDEX}`. You need to create or download ([link](https://utexas.box.com/s/3610huk9fu33m6wic16oe7crx8cahpl8)) an `hdf5`-format dataset file and specify the path to the dataset file as `PATH_TO_CONFIG`.
