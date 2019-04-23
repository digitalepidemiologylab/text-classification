# Kubetext
A supervised text classification framework 

## Install
```
conda env create -f environment.yml
```

## Usage
```
python main.py <command> [<args>]

Available commands:
  split            Splits data into training and test data
  train            Train a classifier based on a config file
  predict          Predict unknown data given a trained model
  generate_config  Generate a config file programmatically
  augment          Augment training data
  fine_tune        Fine-tune pre-trained language models
  learning_curve   Compute learning curve
```

## Example

1) Define a file `config.json` in your root folder.
```
cp example.config.json config.json
```
Content of `config.json`:
```
{
  "runs": [{
    "name": "test_example",
    "model": "bert",
    "overwrite": true,
    "num_epochs": 1
  }],
  "params": {
    "train_data": "example_train.csv",
    "test_data": "example_test.csv"
  }
}
```

2) Train model
```
python main.py train
```
Trained model can be found in `./output/test_example/`
