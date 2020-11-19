"""
Main pipeline helpers
=====================
"""

import os
import sys
import json
import uuid
import logging
import itertools
from dataclasses import asdict
from datetime import datetime
import multiprocessing
from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np
from tqdm import tqdm

import sklearn.model_selection
import joblib

from . import ConfigReader
from .misc import JSONEncoder, get_df_hash
from .nested_dict import flatten_dict, set_nested_value

from .config_manager import Mode


def preprocess(run_config):
    logger = logging.getLogger(__name__)
    try:
        set_label_mapping = __import__(
            'txtcls.models.' + run_config.model.name,
            fromlist=['set_label_mapping']).set_label_mapping
        set_label_mapping(run_config.data.train,
                          run_config.data.test,
                          run_config.path.output)
    except AttributeError:
        logger.info("No 'set_label_mapping'")
    prepare_data = __import__(
        'txtcls.models.' + run_config.model.name,
        fromlist=['prepare_data']).prepare_data
    for v in asdict(run_config.data).values():
        data_path = prepare_data(
            v, run_config.path.output,
            asdict(run_config.preprocess))
        if isinstance(data_path, list):
            data_path = ', '.join(data_path)
        logger.info(f'Prepared data from {v} to {data_path}')


def train(run_config):
    """Trains and evaluates"""
    model = get_model(run_config.model.name)
    logger = logging.getLogger(__name__)
    if getattr(run_config.data, 'augment', None):
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
    test_output = os.path.join(run_config.path.output, 'test_output.json')
    if run_config.write_test_output:
        keys = ['text', 'label', 'prediction']
        df = pd.DataFrame({i: result[i] for i in keys})
        df.to_csv(os.path.join(run_config.path.output, 'test_output.csv'))
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
                    output += '{:<20}: {}\n'.format(
                        new_key, result[new_key])
                else:
                    output += '{:<20}: {:.4f}\n'.format(
                        new_key, result[new_key])
    logger.info(output)
    logger.info(f'Training for model `{run_config.name}` finished. '
                f'Model output written to `{run_config.path.output}`')


def predict(run_config, path=None, data=None, output_cols=[],
            output_folder='predictions', col='text', no_file_output=False,
            in_parallel=False, verbose=False, output_formats=None):

    def read_input_data(path, chunksize=2**15, usecols=[col]):
        if path.endswith('.csv'):
            for text_chunk in pd.read_csv(
                path, usecols=usecols, chunksize=chunksize
            ):
                yield text_chunk[col].tolist()
        elif path.endswith('.txt'):
            with open(path, 'r') as f:
                num_lines = sum(1 for line in f)
                f.seek(0)
                for i, line in enumerate(f.readlines()):
                    text_chunk = []
                    text_chunk.append(line)
                    if i % chunksize == 0 and i > 0:
                        yield text_chunk
        else:
            raise ValueError('Please provide the input file with a file '
                             'extension of either `csv` or `txt`')
    logger = logging.getLogger(__name__)
    model = get_model(run_config.model.name)
    if data is None:
        # Reads from file
        if path is None:
            raise ValueError('Provide either a path or data argument')
        logger.info(f'Reading data from {path}...')
        chunksize = 2**14
        input_data = read_input_data(path, chunksize=chunksize)
        with open(path, 'r') as f:
            num_it = int(sum([1 for _ in f]) / chunksize) + 1
    else:
        # Reads from argument
        input_data = [[data]]
        num_it = len(input_data)
        verbose = True
        run_config.output_attentions = True
    logger.info('Predicting...')
    if in_parallel:
        num_cpus = max(multiprocessing.cpu_count() - 1, 1)
        parallel = joblib.Parallel(n_jobs=num_cpus)
        predict_delayed = joblib.delayed(model.predict)
        output = parallel((
            predict_delayed(run_config, data=predict_data)
            for predict_data in tqdm(
                input_data, total=num_it, unit='chunk',
                disable=bool(path is None))))
        output = list(itertools.chain(*output))
    else:
        output = []
        for predict_data in tqdm(
            input_data, total=num_it, unit='chunk', disable=bool(path is None)
        ):
            predictions = model.predict(run_config, data=predict_data)
            output.extend(predictions)
    if len(output) == 0:
        logger.error('No predictions returned.')
        return
    output_cols = output_cols.split(',')
    if not no_file_output and path is not None:
        if not isinstance(output_formats, list):
            output_formats = ['csv', 'json']
        unique_id = uuid.uuid4().hex[:5]
        for fmt in output_formats:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            output_file = os.path.join(
                output_folder,
                'predicted_{}_{}_{}.{}'.format(
                    run_config.name,
                    datetime.now().strftime('%Y-%m-%d'), unique_id, fmt))
            logger.info('Writing output file {}...'.format(output_file))
            if fmt == 'csv':
                df = pd.DataFrame(output)
                df['label'] = [l[0] for l in df.labels.values]
                df['probability'] = [p[0] for p in df.probabilities.values]
                df[output_cols].to_csv(output_file, index=False)
            elif fmt == 'json':
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=4, cls=JSONEncoder)
    if verbose or (data is not None and path is None):
        logger.info('Prediction output:')
        logger.info(json.dumps(output, indent=4, cls=JSONEncoder))


def pretrain(run_config):
    assert run_config['model'] in ['fasttext', 'bert']
    if run_config['model'] == 'bert':
        if 'use_tf' in run_config and run_config['use_tf']:
            from ..models.finetune_tf_transformer import FinetuneTfTransformer
            model = FinetuneTfTransformer()
        else:
            from ..models.finetune_transformer import FinetuneTransformer
            model = FinetuneTransformer()
    else:
        from ..models.fasttext_pretrain import FastTextPretrain
        model = FastTextPretrain()
    model.init(run_config)
    model.train()

    if 'pretrain_test_data' in run_config:
        result = model.test()
        pretrain_test_output = os.path.join(
            run_config.path.output, 'pretrain_test_output.json')
        with open(pretrain_test_output, 'w') as f:
            json.dump(result, f, cls=JSONEncoder, indent=4)


def get_model(model_name):
    """Dynamically import model class and return model instance"""
    if model_name == 'fasttext':
        from ..models.fasttext import FastText
        return FastText()
    if model_name == 'fasttext_pretrain':
        from ..models.fasttext_pretrain import FastTextPretrain
        return FastTextPretrain()
    if model_name == 'bag_of_words':
        from ..models.bag_of_words import BagOfWordsModel
        return BagOfWordsModel()
    if model_name == 'bert':
        from ..models.bertmodel import BERTModel
        return BERTModel()
    if model_name == 'openai_gpt2':
        from ..models.openai_gpt2 import OpenAIGPT2
        return OpenAIGPT2()
    if model_name == 'dummy':
        from ..models.dummy_models import DummyModel
        return DummyModel()
    if model_name == 'random':
        from ..models.dummy_models import RandomModel
        return RandomModel()
    if model_name == 'weighted_random':
        from ..models.dummy_models import WeightedRandomModel
        return WeightedRandomModel()
    else:
        raise NotImplementedError('Model `{}` is unknown'.format(model_name))


def generate_config(name, model, params, global_params,
                    train_data=None, test_data=None):
    """Generates a grid search config"""
    def _parse_value(s, allow_str=True):
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
        elif s in ['none', 'None']:
            return None
        else:
            if allow_str:
                return s
            else:
                raise ValueError(
                    f"The given parameter value of '{s}' could not "
                    "be converted into int, float or bool")

    logger = logging.getLogger(__name__)

    config = {'globals': {'data': {}}}

    if train_data is None and test_data is None:
        # raise ValueError("Both 'train_data' and 'test_data' are None. "
        #                  "Please provide at least one data path")
        logger.info("No 'train_data' or 'test_data' given. "
                    "Ensure there's data path in the config")
    if train_data is not None:
        config['globals']['data']['train'] = train_data
    if test_data is not None:
        config['globals']['data']['test'] = test_data

    for param in global_params:
        split = param.split(':')
        if len(split) != 2:
            raise ValueError(f'Param {param} has to be of format `key:value`')
        key, value = split
        keys = key.split('.')
        set_nested_value(config['globals'], keys, _parse_value(value, allow_str=True))

    gs_params = []
    for param in params:
        param_split = param.split(':')
        if len(param_split) != 3:
            raise ValueError(
                f'Param {param} has to be of format `key:modifier:values`')
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
    params_lists = [list(i.values())[0] for i in gs_params]
    params_keys = [list(i.keys())[0] for i in gs_params]
    for i, params_combination in enumerate(itertools.product(*params_lists)):
        params = dict(zip(params_keys, params_combination))
        run_config = {}

        for k, v in params.items():
            keys = k.split('.')
            set_nested_value(run_config, keys, v)
        set_nested_value(run_config, ['model', 'name'], model)
        run_config['name'] = '{}_{}'.format(name, i)
        runs.append(run_config)
    if len(runs) == 1:
        # Get rid of integer for a single run
        runs[0]['name'] = runs[0]['name'][:-2]
    config['runs'] = runs

    f_name = 'config.{}.json'.format(name)
    with open(f_name, 'w') as f:
        json.dump(config, f, cls=JSONEncoder, indent=4)
    logger.info(f'Successfully generated file `{f_name}` with {i + 1} runs')


def train_test_split(name, test_size=0.2, label_tags=None,
                     balanced_labels=False, seed=42):
    """Splits cleaned labelled data into training and test set.

    Args:
        test_size (float): Fraction of data which should be reserved
            for test data. Default: 0.2
        label_tags (bool): Only select examples with certain label tags
        balanced_labels (bool): Ensure equal label balance. Default: ``False``
        seed (int): Random seed. Default: 42
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
    train_set, test_set = sklearn.model_selection.train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True)
    for dtype, data in [['train', train_set], ['test', test_set]]:
        f_name = '{}_split_{}_seed_{}{}.csv'.format(
            dtype, int(100 * test_size), seed, flags)
        f_path = os.path.join('data', f_name)
        data.to_csv(f_path, index=None, encoding='utf8')
        logger.info(f'Successfully wrote file {f_path}')


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


def generate_text(**config):
    # TODO: No function ConfigReader().get_default_config()
    # TODO: Do we still need this as a helper? Where to better put it?
    model = get_model(config.get('model', 'openai_gpt2'))
    config_reader = ConfigReader()
    config = config_reader.get_default_config(base_config=config)
    return model.generate_text(config.seed, config)


def learning_curve(config_path):
    from ..utils import LearningCurve
    lc = LearningCurve(config_path)
    lc.init()
    configs = lc.generate_configs()
    for config in configs:
        train(config)


def find_git_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find git root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])


def get_label_mapping(run_path):
    label_mapping_path = os.path.join(run_path, 'label_mapping.pkl')
    if not os.path.isfile(label_mapping_path):
        raise FileNotFoundError(
            f'Could not find label mapping file {label_mapping_path}')
    with open(label_mapping_path, 'rb') as f:
        label_mapping = joblib.load(f)
    return label_mapping


def augment_training_data(
    n=10, min_tokens=8, repeats=1, n_sentences_after_seed=8,
    source='training_data', should_contain_keyword='', verbose=False
):
    raise NotImplementedError


def optimize(config_path):
    from ..utils import Optimize
    opt = Optimize(config_path)
    opt.init()
    opt.run()
