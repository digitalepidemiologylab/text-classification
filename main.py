import argparse
import sys, os
from utils.config_reader import ConfigReader
from utils.helpers import train_test_split, train, predict,generate_config, augment_training_data, fine_tune, generate_fine_tune_input_data, learning_curve
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
  fine_tune        Fine-tune pre-trained language models
  learning_curve   Compute learning curve
"""

class ArgParse(object):
    def __init__(self):
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
        parser = argparse.ArgumentParser(description='Split annotated data into training and test data set')
        parser.add_argument('--question', type=str, required=False, default='sentiment', help='Which data to load (has to be a valid question tag)')
        parser.add_argument('--relevant', dest='relevant', action='store_true', default=False, help='Only get data which was deemed relevant')
        parser.add_argument('--balanced_labels', dest='balanced_labels', action='store_true', default=False, help='Ensure equal label balance')
        parser.add_argument('--all_questions', dest='all_questions', action='store_true', default=False, help='Generate files for all available question tags. This overwrites the `question` argument. Default: False.')
        parser.add_argument('--label_tags', required=False, default=[], nargs='+', help='Only select examples with certain label tags')
        parser.add_argument('--test_size', type=float, required=False, default=0.2, help='Fraction of test size')
        parser.add_argument('--seed', type=int, required=False, default=42, help='Random state split')
        args = parser.parse_args(sys.argv[2:])
        train_test_split(question=args.question, test_size=args.test_size, seed=args.seed, relevant=args.relevant, balanced_labels=args.balanced_labels, all_questions=args.all_questions, label_tags=args.label_tags)

    def train(self):
        """Train model based on config. The following config keys can/should be present in the config file (in runs or params):
        - name (required): Unique name of the run
        - model (required): One of the available models (e.g. fasttext, bert, etc.)
        - overwrite: If run output folder is already present, wipe it and create new folder
        - train_data (required): Path to training data (if only filename is provided it should be located under `data/5_labels_cleaned/`)
        - test_data (required): Path to test data (if only filename is provided it should be located under `data/5_labels_cleaned/`)
        - write_test_output: Write output csv of test evaluation (default: False)
        - embeddings: (fasttext only) Use word embedding file
        - test_only: Runs test file only and skips training (default: False) 
        - parallel: Run in parallel (not recommended for models requiring GPU training)
        - init_checkpoint: (bert only) Start training from pre-existing checkpoint (path to folder)
        """
        parser = argparse.ArgumentParser(description='Train a classifier based on a config file')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        config_reader = ConfigReader()
        config = config_reader.parse_config(args.config)
        if len(config.runs) > 1 and config.parallel:
            num_cpus = os.cpu_count() - 1
            pool = multiprocessing.Pool(num_cpus)
            pool.map(train, config.runs)
        else:
            for run_config in config.runs:
                train(run_config)

    def predict(self):
        parser = argparse.ArgumentParser(description='Predict classes based on a config file and input data')
        parser.add_argument('-r', '--run', metavar='R', required=True, type=str, default=None, help='Name of run')
        parser.add_argument('-p', '--path', metavar='P', type=str, default=None, help='Path of data file for predictions')
        parser.add_argument('-d', '--data', metavar='D', type=str, default=None, help='Input sentence')
        parser.add_argument('-o', '--output', metavar='O', type=str, default=None, help='Path to output file')
        parser.add_argument('-f', '--format', metavar='F', type=str, default='csv', help='Format of prediction output (csv,json)')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output (prediction results)')
        args = parser.parse_args(sys.argv[2:])
        predict(args.run, path=args.path, data=args.data, format=args.format, output=args.output, verbose=args.verbose)

    def generate_config(self):
        parser = argparse.ArgumentParser(description='Generate config for grid search hyperparameter search')
        args = parser.parse_args(sys.argv[2:])
        generate_config()

    def generate_text(self):
        parser = argparse.ArgumentParser(description='Generate text')
        parser.add_argument('-s', '--seed', type=str, required=False, dest='seed', help='Seed text to start generating text from.')
        args = parser.parse_args(sys.argv[2:])
        model = OpenAIGPT2()
        text = model.generate_text(seed_text=args.seed)
        print(text)

    def augment(self):
        parser = argparse.ArgumentParser(description='Augment training data')
        parser.add_argument('-n', '--num', type=int, default=10, required=False, dest='num', help='Number of tweets to generate text from')
        parser.add_argument('-s', '--source', type=str, default='training_data', required=False, dest='source', help='Source for seed data (training data (default), or seed_data)')
        parser.add_argument('-r', '--repeats', type=int, default=1, required=False, dest='repeats', help='Number of repeated times a single seed should be used')
        parser.add_argument('-c', '--contains', type=str, default='crispr', required=False, dest='contains', help='Only collect sentences which contain keyword')
        parser.add_argument('--n_sentences_after_seed', type=int, default=8, required=False, dest='n_sentences_after_seed', help='Consider only first n predicted sentences after seed')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output')
        args = parser.parse_args(sys.argv[2:])
        augment_training_data(n=args.num, min_tokens=8, source=args.source, repeats=args.repeats, should_contain_keyword=args.contains, n_sentences_after_seed=args.n_sentences_after_seed, verbose=args.verbose)

    def fine_tune(self):
        parser = argparse.ArgumentParser(description='Fine-tune language model')
        parser.add_argument('-n', '--name', type=str, required=True, dest='name', help='Name of run')
        parser.add_argument('-m', '--model', type=str, default='bert', required=False, dest='model', help='Model to fine-tune')
        parser.add_argument('--overwrite', default=False, required=False, action='store_true', dest='overwrite', help='Overwrite pre-existing model of same name')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output')
        args = parser.parse_args(sys.argv[2:])
        fine_tune(args.model, args.name, args.overwrite)

    # def generate_fine_tune_input_data(self):
    #     generate_fine_tune_input_data()

    def learning_curve(self):
        parser = argparse.ArgumentParser(description='Generate learning curve')
        parser.add_argument('-c', '--config', metavar='C', required=False, default='config.json', help='Name/path of configuration file. Default: config.json')
        args = parser.parse_args(sys.argv[2:])
        learning_curve(args.config)

if __name__ == '__main__':
    ArgParse()
