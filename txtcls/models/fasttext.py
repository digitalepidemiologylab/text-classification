"""
FastText: text classification
=============================
"""

import os
import csv
import json
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np
import fasttext
import joblib
from munch import DefaultMunch
import swifter

from .base_model import BaseModel, get_default_args
from ..utils.preprocess import preprocess_fasttext
from ..utils.misc import JSONEncoder

tqdm.pandas()
logger = logging.getLogger(__name__)


class FastText(BaseModel):
    """Wrapper for FastText"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.label_mapping = None
        self.train_config = None

    def setup(self, data_path, output_path, train=False):
        if not train:
            if self.model is None:
                self.model = fasttext.load_model(
                    os.path.join(output_path, 'model.bin'))
            if self.train_config is None:
                with open(
                    os.path.join(output_path, 'run_config.json'), 'r'
                ) as f:
                    self.train_config = DefaultMunch.fromDict(json.load(f))
        if self.label_mapping is None:
            with open(os.path.join(data_path, 'label_mapping.json'), 'r') as f:
                self.label_mapping = json.load(f)

    def train(self, config):
        """Trains supervised FastText.
        
        For config params, see https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters
        ```
        input             # training file path (required)
        lr                # learning rate [0.1]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [1]
        minCountLabel     # minimal number of label occurences [1]
        minn              # min length of char ngram [0]
        maxn              # max length of char ngram [0]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [softmax]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        label             # label prefix ['__label__']
        verbose           # verbose [2]
        pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
        ```
        """
        model_path = os.path.join(config.path.output, 'model.bin')
        self.setup(config.path.data, config.path.output, train=True)

        # Prepare data
        # train_data_path = prepare_data(
        #     config.data.train, config.path.output,
        #     dict(config.preprocess), config.model.get('label', '__label__'))
        train_data_path = config.data.train

        # Train model
        logger.info('Training classifier...')
        self.model = fasttext.train_supervised(
            train_data_path, **dict(config.model.params))

        # Save model
        if config.get('save_model', True):
            logger.info('Saving model...')
            self.model.save_model(model_path)
        if config.get('quantize', False):
            logger.info('Quantizing model...')
            self.model.quantize(train_data_path, retrain=True)
            self.model.save_model(model_path)

        # Save model state
        logger.info('Saving params...')
        for k, v in get_default_args(fasttext.train_supervised).items():
            if k not in config.model.params:
                config.model.params[k] = v
        self.add_to_config(
            config.path.output, config.model.params)

    def predict(self, config, data):
        self.setup(config.path.data, config.path.output)
        candidates = self.model.predict(data, k=len(self.label_mapping))
        predictions = [{
            'labels': [
                label[len(self.train_config.model.get('label', '__label__')):]
                for label in candidate[0]],
            'probabilities': candidate[1].tolist()
        } for candidate in zip(candidates[0], candidates[1])]
        return predictions

    def test(self, config):
        self.setup(config.path.data, config.path.output)

        # Preparing data
        logger.info('Reading test data...')
        print(config.data.test)
        df = pd.read_csv(
            config.data.test,
            usecols=['text', 'label'], dtype={'text': str, 'label': str})
        print(df.head())
        test_x, test_y = df['text'].tolist(), df['label'].tolist()
        test_y = [self.label_mapping[y] for y in test_y]

        # Predict and get metrics
        predictions = self.predict(config, test_x)
        y_pred = [self.label_mapping[p['labels'][0]] for p in predictions]
        result_out = self.performance_metrics(
            test_y, y_pred, label_mapping=self.label_mapping)

        if config.write_test_output:
            test_output = self.get_full_test_output(
                y_pred,
                test_y,
                label_mapping=self.label_mapping,
                test_data_path=config.data.test,
                text=test_x)
            result_out = {**result_out, **test_output}
        return result_out


def set_label_mapping(train_data_path, test_data_path, output_path):
    labels = pd.concat([
        pd.read_csv(train_data_path, usecols=['label']),
        pd.read_csv(test_data_path, usecols=['label'])])
    labels = np.unique(labels['label'])
    labels.sort()
    label_mapping = {}
    for i, label in enumerate(np.unique(labels)):
        label_mapping[label] = i
    with open(os.path.join(output_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)


def preprocess_data(data_path, preprocess, preprocessing_config):
    """Reads and preprocesses data.

    Args:
        data_path (str): Path to `.csv` file with two columns,
            ``'text'`` and ``'label'``
        preprocess (bool): Whether to preprocess each text
        preprocessing_config (JSON): Config with args for
            ``text.utils.preprocess``

    Returns:
        df (pandas DataFrame): Dataframe with preprocessed ``'text'`` field
    """
    # Read data
    logger.info(f'Reading data from {data_path}...')
    df = pd.read_csv(
        data_path,
        usecols=['text', 'label'], dtype={'text': str, 'label': str})
    # Drop nans in 'text' or 'label'
    num_loaded = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    # Preprocess data
    if preprocess is True:
        logger.info('Preprocessing data...')
        df['text'] = df.text.swifter.apply(
            preprocess_fasttext, **preprocessing_config)
    # Drop empty strings in 'text'
    df = df[df['text'] != '']
    num_filtered = num_loaded - len(df)
    if num_filtered > 0:
        logger.warning(
            f'Filtered out {num_filtered:,} from {num_loaded:,} samples!')
    return df


def prepare_data(data_path, output_dir_path,
                 preprocess, preprocessing_config,
                 label_prefix):
    """Prepares data for FastText training.

    First, preprocesses data with ``preprocess_data``. Second, formats
    data for FastText.

    Args:
        data_path (str): Path to `.csv` file with two columns,
            ``'text'`` and ``'label'``
        output_dir_path (str): Path to the output folder
        label_prefix (str): Label prefix parameter for supervised FastText

    Returns:
        output_file_path (str): Path to temporary file with
            preprocessed formatted data
    """
    # Preprocess data
    df = preprocess_data(data_path, preprocess, preprocessing_config)
    # Write data
    if 'test' in os.path.basename(data_path) or 'dev' in os.path.basename(data_path):
        # Create paths
        output_file_path = os.path.join(
            output_dir_path, os.path.basename(data_path))
        # Write
        df.to_csv(
            output_file_path,
            index=False, header=True)
        return output_file_path
    # Create paths
    output_file_path = os.path.join(
        output_dir_path, os.path.basename(data_path))
    output_file_path = '.'.join(output_file_path.split('.')[:-1] + ['txt'])
    # Write
    with open(output_file_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f'{label_prefix}{row.label} '
                    f'{row.text}\n')
    return output_file_path
