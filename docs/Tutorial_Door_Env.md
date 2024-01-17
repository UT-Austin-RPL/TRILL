# Tutorial Door Env
This tutorial will walk you through collecting, training, and evaluating a data set for the door environment. This is intended to help users get accustom to the repository and be able to extrapolate this information to new environments.

## Install and Setup
Follow the [installation guide](Setup-Guide.md).

## Collecting the Data
> Want to skip this step? Download our [pre-gathered door environment data set](https://utexas.box.com/s/3610huk9fu33m6wic16oe7crx8cahpl8).

~~ This section is coming soon ~~

## Training

To start training, you can run the training script.
```
python scripts/train.py  --env=door --data=/path/to/datasets/demo250
```

> **Common Issue**: If you are running out of memory, try changing the batch size in the configuration file.

This will generate the checkpoint files inside the `save/` directory.

## Evaluation
> Want to skip this step? Download our [pre-trained door Visuomotor Policies](https://utexas.box.com/s/qn3156sxpejx4zf4piq5zh97srl5zcto).

Evaluation is useful for seeing the status of your visuomotor policy. The following script will run the door opening environment with the generated policy.
```
python3 scripts/evaluate.py --env=door --path=/path/to/checkpoints/door_
```
