# A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION - Baseline Improvement

## Setup

To download the requirements, data, and initial embeddings run:

```sh
./setup.sh
```

## Running experiments

To run an experiment, provide the path to an experiment json. For example:

```sh
python main.py experiments/experiment1.json
```

This will run the experiment for 5 epochs.
The training error is logged to a `training_error.log` file inside the `/models` directory.

## Loading model

The model will be saved at the end of each epoch so you can evaluate the results later.
The params are saved inside the `/models` directory.

You can use the `load_model.py` script to load the validation set and the model after a specific epoch.
For example, the following will load the model used in experiment1 after the 0th (first) epoch:

```sh
python -i load_model.py experiments/experiment1.json 0
```

## Other Baselines
To experiment and/or peruse the other baselines that were implemented, please check out the `baseline-baselines` directory.
