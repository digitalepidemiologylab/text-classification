import argparse
import sys, os
import multiprocessing
import logging

USAGE_DESC = """
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
"""

class ArgParse(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
        parser = argparse.ArgumentParser(
                description='',
                usage=USAGE_DESC)
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        getattr(self, args.command)()

    def split(self):
        from utils.helpers import train_test_split
        parser = argparse.ArgumentParser(description='Split annotated data into training and test data set')
        parser.add_argument('-n', '--name', type=str, required=True, help='Name of dataset or file path')
        parser.add_argument('-s', '--test_size', type=float, required=False, default=0.2, help='Fraction of test size')
        parser.add_argument('--balanced_labels', dest='balanced_labels', action='store_true', default=False, help='Ensure equal label balance')
        parser.add_argument('--label_tags', required=False, default=[], nargs='+', help='Only select examples with certain label tags')
        parser.add_argument('--seed', type=int, required=False, default=42, help='Random state split')
        args = parser.parse_args(sys.argv[2:])
        train_test_split(name=args.name, test_size=args.test_size, balanced_labels=args.balanced_labels, label_tags=args.label_tags, seed=args.seed)

    def train(self):
        """Train model based on config. The following config keys can/should be present in the config file (in runs or params):
        - name (required): Unique name of the run
        - model (required): One of the available models (e.g. fasttext, bert, etc.)
        - overwrite: If run output folder is already present, wipe it and create new folder
        - train_data (required): Path to training data (if only filename is provided it should be located under `data/`)
        - test_data (required): Path to test data (if only filename is provided it should be located under `data/`)
        - augment_data: Path to augment data (if only filename is provided it should be located under `data/`)
        - write_test_output: Write output csv of test evaluation (default: False)
        - test_only: Runs test file only and skips training (default: False) 
        - parallel: Run in parallel (not recommended for models requiring GPU training)
        """
        from utils.helpers import train
        from utils.config_reader import ConfigReader
        parser = argparse.ArgumentParser(description='Train a classifier based on a config file')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        config_reader = ConfigReader()
        config = config_reader.parse_config(args.config)
        if len(config.runs) > 1 and config.params.parallel:
            num_cpus = os.cpu_count() - 1
            pool = multiprocessing.Pool(num_cpus)
            pool.map(train, config.runs)
        else:
            for run_config in config.runs:
                train(run_config)

    def predict(self):
        from utils.helpers import predict
        parser = argparse.ArgumentParser(description='Predict classes based on a config file and input data and output predictions')
        parser.add_argument('-r', '--run', required=True, type=str, default=None, help='Name of run')
        parser.add_argument('-p', '--path', required=False, type=str, default=None, help='Path of data file for predictions')
        parser.add_argument('-d', '--data', required=False, type=str, default=None, help='Input text as argument (ignored if path is given)')
        parser.add_argument('--no_file_output', dest='no_file_output', default=False, action='store_true', help='Do not write output file (default: Write output file to `./predictions/` folder)')
        parser.add_argument('--verbose', dest='verbose', default=False, action='store_true', help='Print predictions')
        args = parser.parse_args(sys.argv[2:])
        predict(args.run, path=args.path, data=args.data, no_file_output=args.no_file_output, verbose=args.verbose)

    def generate_config(self):
        from utils.helpers import generate_config
        parser = argparse.ArgumentParser(description='Generate config for grid search hyperparameter search.')
        parser.add_argument('--name', required=True, type=str, help='Global name prefix and name of output file.')
        parser.add_argument('--train_data', required=True, type=str, help='Train data path')
        parser.add_argument('--test_data', required=True, type=str, help='Test data path')
        parser.add_argument('-m', '--models', required=True, nargs='+', help='List of models. Eeach model will be combined with each param pair.')
        parser.add_argument('-p', '--params', required=False, nargs='*', default=[], help='Arbitrary list of grid search params of the format `key:modifier:values`. \
                Key=hyperparameter name, modifier=Can be either `val` (individual values), `lin` (linspace), or `log` (logspace), followed by the respective values or params for the lin/log space. \
                Examples: num_epochs:val:2,3 or learning_rate:log:-6,-2,4')
        parser.add_argument('-g', '--globals', required=False, nargs='*', default=[], help='List of global params which will be passed to all runs of the format `key:value`')
        args = parser.parse_args(sys.argv[2:])
        generate_config(name=args.name, train_data=args.train_data, test_data=args.test_data, models=args.models, params=args.params, global_params=args.globals)

    def generate_text(self):
        from utils.helpers import generate_text
        parser = argparse.ArgumentParser(description='Generate text')
        parser.add_argument('--seed_text', type=str, required=True, help='Seed text to start generating text from.')
        args, other_args = parser.parse_known_args(sys.argv[2:])
        for arg in other_args:
            if arg.startswith(("-", "--")):
                parser.add_argument(arg)
        args = parser.parse_args(sys.argv[2:])
        text = generate_text(**vars(args))
        print(text)

    def augment(self):
        from utils.helpers import augment_training_data
        parser = argparse.ArgumentParser(description='Augment training data')
        parser.add_argument('-n', '--num', type=int, default=10, required=False, dest='num', help='Number of tweets to generate text from')
        parser.add_argument('-s', '--source', type=str, default='training_data', required=False, dest='source', help='Source for seed data (training data (default), or seed_data)')
        parser.add_argument('-r', '--repeats', type=int, default=1, required=False, dest='repeats', help='Number of repeated times a single seed should be used')
        parser.add_argument('-c', '--contains', type=str, default='', required=False, dest='contains', help='Only collect sentences which contain keyword')
        parser.add_argument('--n_sentences_after_seed', type=int, default=8, required=False, dest='n_sentences_after_seed', help='Consider only first n predicted sentences after seed')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output')
        args = parser.parse_args(sys.argv[2:])
        augment_training_data(n=args.num, min_tokens=8, source=args.source, repeats=args.repeats, should_contain_keyword=args.contains, n_sentences_after_seed=args.n_sentences_after_seed, verbose=args.verbose)

    def fine_tune(self):
        """Finetune model based on config. The following config keys can/should be present in the config file (in runs or params):
        - name (required): Unique name of the run
        - model (required): One of the models which can be finetuned (e.g. bert, etc.)
        - fine_tune_data (required): Path to unannotated data: A csv with a text column (if only filename is provided it should be located under `data/`)
        - overwrite: Wipe existing finetuned model with same name
        - ... all additional model-specific parameters
        """
        from utils.helpers import fine_tune
        from utils.config_reader import ConfigReader
        parser = argparse.ArgumentParser(description='Train a classifier based on a config file')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        config_reader = ConfigReader()
        config = config_reader.parse_fine_tune_config(args.config)
        for run_config in config.runs:
            fine_tune(run_config)

    def learning_curve(self):
        """Compute learning curve for model. The following keys can/should be present in the config file:
        - name (required): Unique prefix of all learning curve runs (each name will be extended with a run index '_run_i')
        - model (required): One of the available models
        - learning_curve_fractions_linspace: Argument to np.linspace which sets the fraction of training data which should be used. Default: [0, 1, 20]
        - learning_curve_repetitions: Number of times each fraction will be trained and evaluated (each fraction will be randomly sampled). Default: 1
        - ... and all parameters which are also available for `train`

        Note that all fractions will contain at least one example of each available class (so that fraction 0 there will be at least num_classes training samples present).
        Output configs will contain keys `learning_curve_fraction` and `learning_curve_num_samples` indicating the portion of training data which was used, as well as a unique
        identifier `learning_curve_id` and a unique run index `learning_curve_index` for each run and a `learning_curve_repetition_index`, being unique for each repetition group.
        """
        from utils.helpers import learning_curve
        parser = argparse.ArgumentParser(description='Generate learning curve')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        learning_curve(args.config)

    def train(self):
        """Train model based on config. The following config keys can/should be present in the config file (in runs or params):
        - name (required): Unique name of the run
        - model (required): One of the available models (e.g. fasttext, bert, etc.)
        - overwrite: If run output folder is already present, wipe it and create new folder
        - train_data (required): Path to training data (if only filename is provided it should be located under `data/`)
        - test_data (required): Path to test data (if only filename is provided it should be located under `data/`)
        - augment_data: Path to augment data (if only filename is provided it should be located under `data/`)
        - write_test_output: Write output csv of test evaluation (default: False)
        - test_only: Runs test file only and skips training (default: False) 
        - parallel: Run in parallel (not recommended for models requiring GPU training)
        """
        from utils.helpers import train
        from utils.config_reader import ConfigReader
        parser = argparse.ArgumentParser(description='Train a classifier based on a config file')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        config_reader = ConfigReader()
        config = config_reader.parse_config(args.config)
        if len(config.runs) > 1 and config.params.parallel:
            num_cpus = os.cpu_count() - 1
            pool = multiprocessing.Pool(num_cpus)
            pool.map(train, config.runs)
        else:
            for run_config in config.runs:
                train(run_config)

    def predict(self):
        from utils.helpers import predict
        parser = argparse.ArgumentParser(description='Predict classes based on a config file and input data and output predictions')
        parser.add_argument('-r', '--run', required=True, type=str, default=None, help='Name of run')
        parser.add_argument('-p', '--path', required=False, type=str, default=None, help='Path of data file for predictions')
        parser.add_argument('-d', '--data', required=False, type=str, default=None, help='Input text as argument (ignored if path is given)')
        parser.add_argument('--no_file_output', dest='no_file_output', default=False, action='store_true', help='Do not write output file (default: Write output file to `./predictions/` folder)')
        parser.add_argument('--verbose', dest='verbose', default=False, action='store_true', help='Print predictions')
        args = parser.parse_args(sys.argv[2:])
        predict(args.run, path=args.path, data=args.data, no_file_output=args.no_file_output, verbose=args.verbose)

    def generate_config(self):
        from utils.helpers import generate_config
        parser = argparse.ArgumentParser(description='Generate config for grid search hyperparameter search.')
        parser.add_argument('--name', required=True, type=str, help='Global name prefix and name of output file.')
        parser.add_argument('--train_data', required=True, type=str, help='Train data path')
        parser.add_argument('--test_data', required=True, type=str, help='Test data path')
        parser.add_argument('-m', '--models', required=True, nargs='+', help='List of models. Eeach model will be combined with each param pair.')
        parser.add_argument('-p', '--params', required=False, nargs='*', default=[], help='Arbitrary list of grid search params of the format `key:modifier:values`. \
                Key=hyperparameter name, modifier=Can be either `val` (individual values), `lin` (linspace), or `log` (logspace), followed by the respective values or params for the lin/log space. \
                Examples: num_epochs:val:2,3 or learning_rate:log:-6,-2,4')
        parser.add_argument('-g', '--globals', required=False, nargs='*', default=[], help='List of global params which will be passed to all runs of the format `key:value`')
        args = parser.parse_args(sys.argv[2:])
        generate_config(name=args.name, train_data=args.train_data, test_data=args.test_data, models=args.models, params=args.params, global_params=args.globals)

    def generate_text(self):
        from utils.helpers import generate_text
        parser = argparse.ArgumentParser(description='Generate text')
        parser.add_argument('--seed_text', type=str, required=True, help='Seed text to start generating text from.')
        args, other_args = parser.parse_known_args(sys.argv[2:])
        for arg in other_args:
            if arg.startswith(("-", "--")):
                parser.add_argument(arg)
        args = parser.parse_args(sys.argv[2:])
        text = generate_text(**vars(args))
        print(text)

    def augment(self):
        from utils.helpers import augment_training_data
        parser = argparse.ArgumentParser(description='Augment training data')
        parser.add_argument('-n', '--num', type=int, default=10, required=False, dest='num', help='Number of tweets to generate text from')
        parser.add_argument('-s', '--source', type=str, default='training_data', required=False, dest='source', help='Source for seed data (training data (default), or seed_data)')
        parser.add_argument('-r', '--repeats', type=int, default=1, required=False, dest='repeats', help='Number of repeated times a single seed should be used')
        parser.add_argument('-c', '--contains', type=str, default='', required=False, dest='contains', help='Only collect sentences which contain keyword')
        parser.add_argument('--n_sentences_after_seed', type=int, default=8, required=False, dest='n_sentences_after_seed', help='Consider only first n predicted sentences after seed')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output')
        args = parser.parse_args(sys.argv[2:])
        augment_training_data(n=args.num, min_tokens=8, source=args.source, repeats=args.repeats, should_contain_keyword=args.contains, n_sentences_after_seed=args.n_sentences_after_seed, verbose=args.verbose)

    def fine_tune(self):
        """Finetune model based on config. The following config keys can/should be present in the config file (in runs or params):
        - name (required): Unique name of the run
        - model (required): One of the models which can be finetuned (e.g. bert, etc.)
        - fine_tune_data (required): Path to unannotated data: A csv with a text column (if only filename is provided it should be located under `data/`)
        - overwrite: Wipe existing finetuned model with same name
        - ... all additional model-specific parameters
        """
        from utils.helpers import fine_tune
        from utils.config_reader import ConfigReader
        parser = argparse.ArgumentParser(description='Train a classifier based on a config file')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        config_reader = ConfigReader()
        config = config_reader.parse_fine_tune_config(args.config)
        for run_config in config.runs:
            fine_tune(run_config)

    def learning_curve(self):
        """Compute learning curve for model. The following keys can/should be present in the config file:
        - name (required): Unique prefix of all learning curve runs (each name will be extended with a run index '_run_i')
        - model (required): One of the available models
        - learning_curve_fractions_linspace: Argument to np.linspace which sets the fraction of training data which should be used. Default: [0, 1, 20]
        - learning_curve_repetitions: Number of times each fraction will be trained and evaluated (each fraction will be randomly sampled). Default: 1
        - ... and all parameters which are also available for `train`

        Note that all fractions will contain at least one example of each available class (so that fraction 0 there will be at least num_classes training samples present).
        Output configs will contain keys `learning_curve_fraction` and `learning_curve_num_samples` indicating the portion of training data which was used, as well as a unique
        identifier `learning_curve_id` and a unique run index `learning_curve_index` for each run and a `learning_curve_repetition_index`, being unique for each repetition group.
        """
        from utils.helpers import learning_curve
        parser = argparse.ArgumentParser(description='Generate learning curve')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        learning_curve(args.config)

    def optimize(self):
        """Performs hyperparameter optimization (requires hypteropt package)
        - name (required): Unique prefix of all learning curve runs (each name will be extended with a run index '_run_i')
        - model (required): One of the available models
        - ... and all parameters which are also available for `train`
        """
        from utils.helpers import optimize
        parser = argparse.ArgumentParser(description='Perform hyperparamter search')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        optimize(args.config)

    def ls(self):
        from utils.list_runs import ListRuns
        parser = argparse.ArgumentParser(description='List trained models')
        parser.add_argument('-m', '--model', type=str, default=None, required=False, dest='model', help='Only show certain models')
        parser.add_argument('-f', '--filename-pattern', type=str, default=None, required=False, dest='filename_pattern', help='Filter by name of training data input file')
        parser.add_argument('-p', '--params', type=str, nargs='+', default=None, required=False, help='Display certain hyperparameters instead of default ones')
        parser.add_argument('-a', '--averaging', type=str, choices=['micro', 'macro', 'weighted'], default='macro', required=False, help='Precision/recall/f1 averaging mode')
        parser.add_argument('--metrics', type=str, nargs='+', default=None, choices=['accuracy', 'f1', 'precision', 'recall'], required=False, help='Metrics to display')
        parser.add_argument('--pattern', type=str, default=None, required=False, dest='pattern', help='Filter run names by pattern')
        parser.add_argument('--names-only', dest='names_only', action='store_true', help='Only list names')
        args = parser.parse_args(sys.argv[2:])
        ls = ListRuns()
        ls.list_runs(pattern=args.pattern, model=args.model, names_only=args.names_only, filename_pattern=args.filename_pattern,
                averaging=args.averaging, metrics=args.metrics, params=args.params)

if __name__ == '__main__':
    ArgParse()
