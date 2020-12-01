# `txcl`: Crowdbreaks Text Classification Tool
A simple supervised text classification CLI framework.


## Installation
### Developer
```bash
git clone https://github.com/crowdbreaks/text-classification.git
cd text-classification
pip install -e .
```

### User
```bash
pip install git+https://github.com/crowdbreaks/text-classification.git
```

Note: You may need to install additional packages for full functionality.


## CLI
For a list of available commands, run
```bash
txcl --help
```
Output:
```
usage: txcl [-h] {main,deploy,plot,print} ...

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

To access a help message for a specific subcommand, run, e.g.
```bash
txcl main train --help
```
Output:
```
usage: txcl main train [-h] [-c C] [--parallel]

Trains model based on config.

optional arguments:
  -h, --help        show this help message and exit
  -c C, --config C  name/path of configuration file (default: config.json)
  --parallel        run in parallel (only recommended for CPU-training) (default: False)
```


## Configuration files
Many of the commands require a path to a configuration (config) file in JSON format.
A config file defines a list of 'runs', each of which specifies a set of parameters that will be used to run your command of choice.

### Parameters merging
The general structure of the config file looks like this

```
{
    "globals": { global_params },
    "runs": [
        { params_run_1 },
        { params_run_2 }
    ]
}
```

Put all the parameters necessary for each run of your command under the `"runs"` key.

If you have parameters that should be fixed for all runs, put them under the `"globals"` key. For example, you might want to preprocess the same data in a number of different ways or train a model with a few fixed parameters, and a few varied ones. In this case, you can put the data path or the fixed model params under `"globals"`.

In summary,
- `"globals"`
    - Whichever parameters you put in here, will be merged with each of the runs' parameters
- `"runs"`
    - These are your run-specific parameters


### List of available parameters

- **path** *(strings, paths)*
    - data
        - *default `'./data'`*
    - output
        - *default `'./output'`*
    - tmp
        - *default `'./tmp'`*
    - other
        - *default `'./other'`*
- **data** *(strings, paths, default `None`)*
    - train
    - val
    - test
- **folders**
    - *string, mode name, default `'new'`:*
        - new
            - Create new folders (output folder, run folders), throw an error in case of existing folders
        - overwrite
            - Create new folders, overwrite existing folders
        - existing
            - Use existing folders, throw an error if no folder found
- **name**
    - *string, [folder] name of the run*
- **preprocess**
    - *dictionary, preprocessing parameters:*
        ```
        standardize_func_name: str = 'standardize'
        min_num_tokens: int = 0
        min_num_chars: int = 0
        lower_case: bool = False
        asciify: bool = False
        remove_punctuation: bool = False
        standardize_punctuation: bool = False
        asciify_emoji: bool = False
        remove_emoji: bool = False
        replace_url_with: Union[str, None] = None
        replace_user_with: Union[str, None] = None
        replace_email_with: Union[str, None] = None
        lemmatize: bool = False
        remove_stop_words: bool = False
        ```
- **model**
    - name
        - *string, model's name:*
            - fasttext
            - bert
    - params
        - *dictionary, model parameters (check separately in the docs for each model)*
    - ...other model-specific parameters (check separately in the docs for each model)
- **test_only**
    - *boolean, default `False`, `txcl main train`-specific* 
- **write_test_output**
    - *boolean, default `False`, `txcl main train`-specific* 

### Required and optional parameters for different commands
**required**, *optional*

- `txcl main preprocess`
    - **name**
    - *path* (default)
    - **data**
        - **train or val, or test**
    - *preprocess* (default)
    - **model**
        - **name**

- `txcl main train`
    - **name**
    - *path* (default)
    - **data**
        - **train**
            - Not required if *test_only* is True
        - *val*
        - **test**
    - *preprocess*
        - Retrieved automatically if the data folder is a preprocessing run
    - **model**
        - **name**
        - **params**
    - *test_only* (default)
    - *write_test_output* (default)

- `txcl main predict`
    - **name**
    - *path* (default)
    - *data* (default)
    - **model**
        - **name**


## Grid configs generation
To generate a config file that contains multiple runs following a parameter grid, use `txcl main generate-config`.

For example,
```bash
txcl main generate-config \
    --mode preprocess \
    --name standardize_ag-news \
    --train-data './data/ag-news/train.csv' \
    --test-data './data/ag-news/dev.csv' \
    --model fasttext \
    --globals 'folders:overwrite' \
              'preprocess.standardize_func_name:standardize' \
    --params 'preprocess.lower_case:val:true,false' \
             'preprocess.remove_punctuation:val:true,false'
```
will generate a config file with 4 runs with varying `'lower_case'` and `'remove_punctuation'` preprocessing keys.

To learn more about this command, run
```bash
txcl main generate-config --help
```


## Example
In this example, you will train a FastText classifier on the [AG's News Topic Classification Dataset](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) example data.

1. Go to `path/to/text-classification/test` on your machine
```bash
cd path/to/text-classification/test
```

2. Check out the data
```bash
head -n 10 data/ag-news/all.csv
```

You should see something like this

label | text
----- | ----
2 | Finish line is in sight for Martin
1 | Blair backs India's quest for permanent seat on UN Security Council (AFP)
3 | "Ford Expands Recall of SUVs to 600,000"

The dataset contains 4 types of labels:
- 1 = World
- 2 = Sports
- 3 = Business
- 4 = Sci/Tech

Note: It is important that the CSV files (train, validation and test) have columns named `text` and `label`.

3. Check out the configs (preprocessing and training)
```bash
cat configs/cli/config.preprocess.ag-news.json
cat configs/cli/config.train.ag-news.json
```

4. Preprocess the data
```bash
txcl main preprocess -c configs/cli/config.preprocess.ag-news.json
```

You can find your preprocessed data along with an exhaustive config file and label mapping in `.output/preprocess_standardize_anonymize`.

5. Train a model
Running this command trains and then automatically evaluates the model on the test set. If no `-c` option is given, it will look for a file called `config.json` in the current folder.

```bash
txcl main train -c configs/cli/config.train.ag-news.json
```

The trained model's artefacts, performance scores and run logs can be found in `./output/train_fasttext_default`.

6. View the results
After training you can enter your output folder and run the list runs command
```bash
cd output
txcl main ls
```
to get a list of all trained models.

Output:
```
List runs
---------

- Pattern:           * 

                        f1_macro  precision_macro  recall_macro  accuracy
name                                                                     
train_fasttext_default  0.844657         0.844617      0.844838  0.844906
```

Note: `generate-config` is a work in progress, [Olesia](https://github.com/utanashati) is going to rewrite this soon.


## Contribute
Feel free to add new text classification models to this. All trained models inherit from a `BaseModel` class defined under `models/`. It contains a blueprint of which methods any new model should implement.
