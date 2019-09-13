# Crowdbreaks text classification 
A simple supervised text classification framework.

## Install
```
conda env create -f environment.yml
```

The environment (Python version 3.6) contains the following packages:
```
- pytorch
- pytorch-transformers
- pandas
- tqdm
- numpy
- munch
- scikit-learn
- visdom
- dill
- nltk
```


## Usage
For a list of available commands run 
```
$ python main.py --help
```
Output:
```
usage: 
python main.py <command> [<args>]

Available commands:
  split            Splits data into training and test data
  train            Train a classifier based on a config file
  predict          Predict unknown data given a trained model
  generate_config  Generate a config file programmatically
  augment          Augment training data
  generate_text    Generate text
  fine_tune        Fine-tune pre-trained language models
  learning_curve   Compute learning curve
  optimize         Perform hyperparameter optimization
  ls               List trained models and performance
```

If you need help to a specific subcommand you can run e.g.
```
python main.py train --help
```
Output:
```
Train a classifier based on a config file

optional arguments:
  -h, --help        show this help message and exit
  -c C, --config C  Name/path of configuration file. Default: config.json
```


## Example
In this example you will train a BERT classifier from IMDB movie review example data.

1) Add some example data to the `data` folder
```
cp other/example_data/example*.csv data/
```
The example CSV data looks like this

text | label 
---- | -----
hide new secretions from the parental units | 0 |
contains no wit , only labored gags |  0 |
that loves its characters and communicates something rather beautiful about human nature | 1 |

It is important that the CSV files (train and test) have a column named `text` and one which is named `label`.

2) Define a file `config.json` in your root folder.
```
cp other/example_data/example*.csv data/
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

3) Train model
This command will train and then automatically evaluate the model on the test set.
```
python main.py train
```
Trained model can be found in `./output/test_example/`

4) View results of models
After training you can run 
```
python main.py ls
```
to get a list of all models trained

# Contribute
Anyone is free to add new text classification models to this. All trained models inherit from a `BaseModel` class defined under `models/`. It contains a blueprint of which functions any new model should have.
