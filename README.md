# Crowdbreaks text classification 
A simple supervised text classification framework.

## Install
```
pip install -r requirements.txt
```
Note: You may need to install additional packages for full functionality.

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
In this example you will train a FastText classifier from the [Sentiment 140](http://help.sentiment140.com/for-students/) example Twitter data.

1) Add some example data to the `data` folder
```
cp other/example_data/example*.csv data/
```
The example CSV data looks like this

text | label 
---- | -----
Waiting for the set with Bumble Bee and Sam figure. Have a little Shia in a little Bumble Bee. | 4 |
@<user> there is never a pot of gold at the end on a rainbow though!   stupid lepercans...their probaly not even real. HAHA | 0 |
  
The dataset contains 2 types of labels 0=negative and 4=positive. It is important that the CSV files (train and test) have a column named `text` and one which is named `label`. All user handles have been replaced by `@<user>`.

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
    "model": "fasttext",
    "overwrite": true,
    "num_epochs": 1
  }],
  "params": {
    "train_data": "example_train.csv",
    "test_data": "example_test.csv"
  }
}
```
The file can contain multiple training & testing rounds (runs) with different sets of parameters. Parameters defined in `params` are global to all runs. Each run needs to contain a unique `name` and needs to have a `model` parameter. The `overwrite` parameter will overwrite an exisiting model with the same name.

3) Train model

This command will train and then automatically evaluate the model on the test set. If not `-c` option is given, train will look for a file called `config.json` in the project root.
```
python main.py train
```
Trained model can be found in `./output/test_example/`

4) View results of models

After training you can run 
```
python main.py ls
```
Output:
```
                 model  num_epochs  f1_macro  precision_macro  recall_macro  accuracy
name                                                                                 
test_example  fasttext          20  0.175074         0.118712      0.333333  0.356137
```

to get a list of all models trained. 

## Contribute
Feel free to add new text classification models to this. All trained models inherit from a `BaseModel` class defined under `models/`. It contains a blueprint of which methods any new model should implement.
