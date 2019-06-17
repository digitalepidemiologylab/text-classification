from utils.config_reader import ConfigReader
from utils.learning_curve import LearningCurve
from utils.misc import JSONEncoder, get_df_hash
import pandas as pd
import numpy as np
import json
import os
import glob
from multiprocessing import Pool
import string
import unicodedata
import re
from tqdm import tqdm
import csv
from collections import Counter
import sklearn.model_selection
import logging
import itertools
from datetime import datetime
import uuid


def train(run_config):
    model = get_model(run_config.model)
    logger = logging.getLogger(__name__)
    if run_config.augment_data:
        logger.info('Augmenting training data')
        run_config = augment(run_config)
    # train
    logger.info('\n\nStart training model for `{}`'.format(run_config.name))
    if not run_config.test_only:
        model.train(run_config)
    # test
    logger.info("\n\nTest results for `{}`:\n".format(run_config.name))
    output = '\n'
    result = model.test(run_config)
    if result is None:
        logger.warning('No test results generated.')
        return
    test_output = os.path.join(run_config.output_path, 'test_output.json')
    if run_config.write_test_output:
        keys = ['text', 'label', 'prediction']
        df = pd.DataFrame({i: result[i] for i in keys})
        df.to_csv(os.path.join(run_config.output_path, 'test_output.csv'))
        for k in keys:
            result.pop(k, None)
    with open(test_output, 'w') as outfile:
        json.dump(result, outfile, cls=JSONEncoder, indent=4)
    print_keys = ['accuracy', 'recall', 'precision', 'f1', 'test_loss', 'loss']
    for key in print_keys:
        for modifier in ['', '_micro', '_macro', '_binary']:
            new_key = key + modifier
            if new_key in result:
                if isinstance(result[new_key], str):
                    output += "{:<20}: {}\n".format(new_key, result[new_key])
                else:
                    output += "{:<20}: {:.4f}\n".format(new_key, result[new_key])
    logger.info(output)
    logger.info("Training for model `{}` finished. Model output written to `{}`".format(run_config.name, run_config.output_path))

def predict(run_name, path=None, data=None, no_file_output=False, verbose=False, output_formats=None):
    def read_input_data(path, chunksize=2**15, usecols=['text']):
        if path.endswith('.csv'):
            for text_chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
                yield text_chunk['text'].tolist()
        elif path.endswith('.txt'):
            with open(path, 'r') as f:
                num_lines = sum(1 for line in datafile)
                datafile.seek(0)
                for i, line in enumerate(f.readlines()):
                    text_chunk = []
                    text_chunk.append(line)
                    if i % chunksize == 0 and i > 0:
                        yield text_chunk
        else:
            raise ValueError('Please provide the input file with a file extension of either `csv` or `txt`.')
    # parse run config
    config_reader = ConfigReader()
    config_path = os.path.join('output', run_name, 'run_config.json')
    config = config_reader.parse_config(config_path, predict_mode=True)
    run_config = config.runs[0]
    logger = logging.getLogger(__name__)
    model = get_model(run_config.model)
    if data is None:
        if path is None:
            raise ValueError('Provide either a path or data argument')
        chunksize = 2**14
        input_data = read_input_data(path, chunksize=chunksize)
        with open(path, 'r') as f:
            num_it = int(sum([1 for _ in f]) / chunksize) + 1
    else:
        input_data = [[data]]
        num_it = len(input_data)
        verbose = True
    logger.info('Predicting...')
    output = []
    for predict_data in tqdm(input_data, total=num_it, unit='chunk', disable=bool(path is None)):
        predictions = model.predict(run_config, data=predict_data)
        output.extend(predictions)
    if len(output) == 0:
        logger.error('No predictions returned.')
        return
    if not no_file_output and path is not None:
        if output_formats is None:
            output_formats = ['csv', 'json']
        unique_id = uuid.uuid4().hex[:5]
        for fmt in output_formats:
            output_path = os.path.join('.', 'predictions')
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            output_file = os.path.join(output_path, 'predicted_{}_{}_{}.{}'.format(run_config.name, datetime.now().strftime('%Y-%m-%d'), unique_id, fmt))
            logger.info('Writing output file {}...'.format(output_file))
            if fmt == 'csv':
                df = pd.DataFrame(output)
                df['label'] = [l[0] for l in df.labels.values]
                df['probability'] = [p[0] for p in df.probabilities.values]
                df.to_csv(output_file, index=False)
            elif fmt == 'json':
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=4, cls=JSONEncoder)
    if verbose or (data is not None and path is None):
        logger.info('Prediction output:')
        logger.info(json.dumps(output, indent=4, cls=JSONEncoder))

def get_model(model_name):
    """Dynamically import model module and return model instance"""
    if model_name == 'fasttext':
        from models.fasttextmodel import FastTextModel
        return FastTextModel()
    elif model_name == 'bag_of_words':
        from models.bag_of_words import BagOfWordsModel
        return BagOfWordsModel()
    elif model_name == 'bert':
        from models.bertmodel import BERTModel
        return BERTModel()
    elif model_name == 'openai_gpt2':
        from models.openai_gpt2 import OpenAIGPT2
        return OpenAIGPT2()
    elif model_name == 'dummy':
        from models.dummy_models import DummyModel
        return DummyModel()
    elif model_name == 'random':
        from models.dummy_models import RandomModel
        return RandomModel()
    elif model_name == 'weighted_random':
        from models.dummy_models import WeightedRandomModel
        return WeightedRandomModel()
    else:
        raise NotImplementedError('Model `{}` is unknown'.format(model_name))

def generate_config(name, train_data, test_data, models, params, global_params):
    """Generates a grid search config"""
    def _parse_value(s, allow_str=False):
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        if s in ['true', 'True']:
            return True
        elif s in ['false', 'False']:
            return False
        else:
            if allow_str:
                return s
            else:
                raise ValueError('The given parameter value of `{}` could not be converted into int, float or bool')
    logger = logging.getLogger(__name__)
    config = {"params": {"train_data": train_data, "test_data": test_data}}
    for g in global_params:
        g = g.split(':')
        if len(g) != 2:
            raise ValueError('Param {} has to be of format `key:value`'.format(param))
        key, value = g
        config['params'][key] = _parse_value(value, allow_str=True)
    gs_params = []
    for param in params:
        param_split = param.split(':')
        if len(param_split) != 3:
            raise ValueError('Param {} has to be of format `key:modifier:values`'.format(param))
        key, modifier, values = param_split
        values = list(map(_parse_value, values.split(',')))
        if modifier == 'val':
            gs_params.append({key: values})
        elif modifier == 'lin':
            gs_params.append({key: np.linspace(*values).tolist()})
        elif modifier == 'log':
            gs_params.append({key: np.logspace(*values).tolist()})
        else:
            raise ValueError('Modifier {} is not recognized.'.format(modifier))
    runs = []
    params = [list(i.values())[0] for i in gs_params] + [models]
    param_names = [list(i.keys())[0] for i in gs_params] + ['model']
    for i, param in enumerate(itertools.product(*params)):
        param_dict = dict(zip(param_names, param))
        param_dict['name'] = '{}_{}'.format(name, i)
        runs.append(param_dict) 
    if len(runs) == 1:
        # get rid of integer for single run
        runs[0]['name'] = runs[0]['name'][:-2]
    config['runs'] = runs
    f_name = 'config.{}.json'.format(name)
    with open(f_name, 'w') as f:
        json.dump(config, f, cls=JSONEncoder, indent=4)
    logger.info('Successfully generated file `{}` with {} runs'.format(f_name, i+1))

def train_test_split(name, test_size=0.2, label_tags=None, balanced_labels=False, seed=42):
    """Splits cleaned labelled data into training and test set
    :param test_size: Fraction of data which should be reserved for test data, default: 0.2
    :param label_tags: Only select examples with certain label tags
    :param balanced_labels: Ensure equal label balance
    :param seed: Random seed (default: 42)
    """
    def get_data(name):
        try:
            return pd.read_csv(os.path.join(name))
        except FileNotFoundError:
            return pd.read_csv(os.path.join('output', name))
    def filter_for_label_balance(df):
        """Performs undersampling for overrepresanted label classes"""
        counts = Counter(df['label'])
        min_count = min(counts.values())
        _df = pd.DataFrame()
        for l in counts.keys():
            _df = pd.concat([_df, df[df['label'] == l].sample(min_count)])
        return _df
    df = get_data(name)
    logger = logging.getLogger(__name__)
    if balanced_labels:
        df = filter_for_label_balance(df)
    flags = '{}'.format('_balanced' if balanced_labels else '')
    train, test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    for dtype, data in [['train', train], ['test', test]]:
        f_name = '{}_split_{}_seed_{}{}.csv'.format(dtype, int(100*test_size), seed, flags)
        f_path = os.path.join('data', f_name)
        data.to_csv(f_path, index=None, encoding='utf8')
        logger.info('Successfully wrote file {}'.format(f_path))

def augment(run_config):
    df_augment = pd.read_csv(run_config.augment_data)
    try:
        df_augment = df_augment[['text', 'label']]
    except KeyError:
        raise Exception('Augmented data needs to have a text and label column.')
    df_train = pd.read_csv(run_config.train_data)
    # concatenate
    df_train = pd.concat([df_train, df_augment], sort=False)
    # check if hash exists
    f_name = '{}.csv'.format(get_df_hash(df_train))
    f_path = os.path.join(run_config.tmp_path, 'augment', f_name)
    if not os.path.isdir(os.path.dirname(f_path)):
        os.makedirs(os.path.dirname(f_path))
    if not os.path.isdir(f_path):
        # shuffle and store
        df_train = df_train.sample(frac=1)
        df_train.to_csv(f_path, index=False)
    # change paths
    run_config.train_data = f_path
    return run_config

def fine_tune(run_config):
    model = get_model()
    model.fine_tune(run_config)

def generate_text(**config):
    model = get_model(config.get('model', 'openai_gpt2'))
    config_reader = ConfigReader()
    config = config_reader.get_default_config(base_config=config)
    return model.generate_text(config.seed_text, config)

def learning_curve(config_path):
    lc = LearningCurve(config_path)
    lc.init()
    configs = lc.generate_configs()
    for config in configs:
        train(config)

def find_project_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])

def augment_training_data(n=10, min_tokens=8, repeats=1, n_sentences_after_seed=8, source='training_data', should_contain_keyword='', verbose=False):
    raise NotImplementedError
