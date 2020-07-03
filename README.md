# Crowdbreaks text classification 
A simple supervised text classification framework.

## Install
```bash
pip install -e .
```
Note: You may need to install additional packages for full functionality.

## Usage
For a list of available commands run 
```bash
$ txtcls --help
```
Output:
```
usage: txtcls [-h] {main,deploy,plot,print} ...

positional arguments:
  {main,deploy,plot,print}
                        sub-commands
    main                main pipeline
    deploy              deployment
    plot                plotting
    print               printing

optional arguments:
  -h, --help            show this help message and exit
```

If you need help to a specific subcommand you can run e.g.
```bash
txtcls main train --help
```
Output:
```
usage: txtcls main train [-h] [-c C] [--parallel]

Trains model based on config.

optional arguments:
  -h, --help        show this help message and exit
  -c C, --config C  name/path of configuration file (default: config.json)
  --parallel        run in parallel (only recommended for CPU-training) (default: False)
```


## Example
In this example you will train a FastText classifier from the [Sentiment 140](http://help.sentiment140.com/for-students/) example Twitter data.

1) Add some example data to the `data` folder
```bash
cp other/example_data/example*.csv data/
```
The example CSV data looks like this

text | label 
---- | -----
Waiting for the set with Bumble Bee and Sam figure. Have a little Shia in a little Bumble Bee. | 4 |
@user there is never a pot of gold at the end on a rainbow though!   stupid lepercans...their probaly not even real. HAHA | 0 |
  
The dataset contains 2 types of labels 0=negative and 4=positive. It is important that the CSV files (train and test) have a column named `text` and one which is named `label`. All user handles have been replaced by `@user`.

2) Define a file `config.json` in your root folder.
```bash
cp other/example_data/example*.csv data/
cp example.config.json config.json
```
Content of `config.json`:
```json
{
  "params": {
    "train_data": "example_train.csv",
    "test_data": "example_test.csv"
  },
  "runs": [{
    "name": "test_example",
    "model": "fasttext",
    "overwrite": true,
    "num_epochs": 1
  }]
}
```
The JSON file consists of two keys:
1. `params`: These are global options/parameters for all runs.
2. `runs`: An array of runs. Each run is a single round of training and testing. Each run needs to have a unique `name` parameter (in this case `test_example`) and needs to specify type type of model used with the `model` parameter.

In the example above additionally `overwrite` was specified t overwrite if there is an existing run with the same name.

3) Train model

This command will train and then automatically evaluate the model on the test set. If no `-c` option is given, train will look for a file called `config.json` in the project root.
```bash
txtcls main train
```

Alternatively, specificy a config file:
```bash
txtcls main train -c my_config_file.json
```

The trained model artefacts, performance scores and run logs can be found in `./output/test_example/`

4) View results of models

After training you can run 
```bash
txtcls main ls
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
