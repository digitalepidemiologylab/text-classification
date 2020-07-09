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
import fasttext
from munch import DefaultMunch

from .base_model import BaseModel, get_default_args
from ..utils.preprocess import preprocess, get_preprocessing_config
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

    def setup(self, output_path, train=False):
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
            self.label_mapping = self.load_label_mapping(output_path)

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
        model_path = os.path.join(
            config.output_path, 'model.bin')
        self.label_mapping = self._set_label_mapping(
            config.train_data, config.test_data, config.output_path)
        self.setup(config.output_path, train=True)

        # Prepare data
        train_data_path = prepare_data(
            config.train_data, config.output_path,
            dict(config.preprocess), config.model.get('label', '__label__'))

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

        print(self.model.predict('politician'))

        # Save model state
        logger.info('Saving params...')
        for k, v in get_default_args(fasttext.train_supervised).items():
            if k not in config.model.params:
                config.model.params[k] = v
        self.add_to_config(
            config.output_path, config.model.params, config.preprocess)

    def predict(self, config, data):
        self.setup(config.output_path)
        data = ['' if pd.isna(d) else d for d in data]
        data = [
            preprocess(d, **dict(self.train_config.preprocess)) for d in data]
        candidates = self.model.predict(data, k=len(self.label_mapping))
        predictions = [{
            'labels': [
                label[len(self.train_config.model.get('label', '__label__')):]
                for label in candidate[0]],
            'probabilities': candidate[1].tolist()
        } for candidate in zip(candidates[0], candidates[1])]
        return predictions

    def test(self, config):
        self.setup(config.output_path)

        # Preparing data
        logger.info('Reading test data...')
        df = preprocess_data(
            config.test_data, dict(self.train_config.preprocess))
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
                test_data_path=config.test_data,
                text=test_x)
            result_out = {**result_out, **test_output}
        return result_out


def preprocess_data(data_path, preprocess_config):
    """Reads and preprocesses data.

    Args:
        data_path (str): Path to `.csv` file with two columns,
            ``'text'`` and ``'label'``
        preprocess_config (JSON): Config with args for
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
    logger.info('Preprocessing data...')
    df['text'] = df.text.progress_apply(preprocess, **preprocess_config)
    # Drop empty strings in 'text'
    df = df[df['text'] != '']
    num_filtered = num_loaded - len(df)
    if num_filtered > 0:
        logger.warning(
            f'Filtered out {num_filtered:,} from {num_loaded:,} samples!')
    return df

def preprocess(self):
    # hash dataset in a temp directory?

def prepare_data(data_path, output_dir_path, preprocess_config, label_prefix):
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
    # Create paths
    output_file_path = os.path.join(
        output_dir_path, os.path.basename(data_path) + '.fasttext.tmp')
    # Read data
    df = preprocess_data(data_path, preprocess_config)
    # Prepare data and writes to temporary file
    with open(output_file_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f'{label_prefix}{row.label} '
                    f'{row.text}\n')
    return output_file_path
