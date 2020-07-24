"""
FastText: word embeddings
=========================
"""

import os
import sys
import logging
import multiprocessing

from tqdm import tqdm
import pandas as pd
import fasttext
import swifter
from joblib import Parallel, delayed, wrap_non_picklable_objects

from .base_model import BaseModel, get_default_args
from .fasttext import preprocess_fasttext

tqdm.pandas()
logger = logging.getLogger(__name__)


class FastTextPretrain(BaseModel):
    """Wrapper for FastText"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.preprocess_config = None

    def train(self, config):
        """Pretrains FastText word representations.

        For config params, see https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
        ```
        input             # training file path (required)
        model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
        lr                # learning rate [0.05]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [5]
        minn              # min length of char ngram [3]
        maxn              # max length of char ngram [6]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [ns]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        verbose           # verbose [2]
        ```
        """

        model_path = os.path.join(config.path.output, 'model.bin')
        vectors_path = os.path.join(config.path.output, 'vectors.vec')

        # Train model
        logger.info('Training model...')
        self.model = fasttext.train_unsupervised(
            config.data.train, **dict(config.model.params))
        if config.get('quantize', False):
            logger.info('Quantizing model...')
            self.model.quantize(config.data.train, retrain=True)

        unsupervised_default = {
            'model': "skipgram",
            'lr': 0.05,
            'dim': 100,
            'ws': 5,
            'epoch': 5,
            'minCount': 5,
            'minCountLabel': 0,
            'minn': 3,
            'maxn': 6,
            'neg': 5,
            'wordNgrams': 1,
            'loss': "ns",
            'bucket': 2000000,
            'thread': multiprocessing.cpu_count() - 1,
            'lrUpdateRate': 100,
            't': 1e-4,
            'label': "__label__",
            'verbose': 2,
            'pretrainedVectors': "",
            'seed': 0,
            'autotuneValidationFile': "",
            'autotuneMetric': "f1",
            'autotunePredictions': 1,
            'autotuneDuration': 60 * 5,  # 5 minutes
            'autotuneModelSize': "",
        }

        # Save model state
        logger.info('Saving params...')
        for k, v in unsupervised_default.items():
            if k not in config.model.params:
                config.model.params[k] = v
        self.add_to_config(config.path.output, config)

        # Save model
        if config.get('save_model', True):
            logger.info('Saving model...')
            self.model.save_model(model_path)
        if config.get('save_vec', True):
            logger.info('Saving vectors...')
            self.save_vec(vectors_path)
            # config.model.params.get('thread', max(os.cpu_count() - 1, 1)),
            # config.model.params.get('verbose', 2))

    def test(self, config):
        pass

    def predict(self, config, data):
        pass

    def save_vec(self, output_vectors_path):
        # https://stackoverflow.com/questions/58337469/how-to-save-fasttext-model-in-vec-format
        # Cannot be parallelized because you can't pickle fasttext model
        # Get all words from model
        words = self.model.get_words()

        with open(output_vectors_path, 'w+') as f:
            # The first line must contain number of total words and
            # vector dimension
            f.write(
                str(len(words)) + ' ' + str(self.model.get_dimension()) + '\n')

            # Line by line, you append vectors to VEC file
            for w in words:
                v = self.model.get_word_vector(w)
                vstr = ''
                for vi in v:
                    vstr += ' ' + str(vi)
                f.write(w + vstr + '\n')


def preprocess_data(data_path, preprocessing_config):
    """Reads and preprocesses data.

    Args:
        data_path (str): Path to `.txt` file
        preprocess (bool): Whether to preprocess each text
        preprocessing_config (JSON): Config with args for
            ``text.utils.preprocess``

    Returns:
        df (pandas DataFrame): Dataframe with preprocessed ``text`` field
    """
    # Read data
    logger.info(f'Reading data from {data_path}...')
    df = pd.read_parquet(data_path)
    # print(df.columns)
    # print(df.lang.value_counts())
    df = df[['user.description']]
    df = df.rename(columns={'user.description': 'text'})
    df = df.reset_index(drop=True)
    # Drop nans
    num_loaded = len(df)
    df.dropna(subset=['text'], inplace=True)
    # Preprocess data
    try:
        standardize_func_name = preprocessing_config['standardize_func_name']
        del preprocessing_config['standardize_func_name']
    except KeyError:
        standardize_func_name = None
    if standardize_func_name is not None:
        logger.info('Standardizing data...')
        standardize_func = getattr(
            __import__(
                'txtcls.utils.standardize',
                fromlist=[standardize_func_name]),
            standardize_func_name)
        df['text'] = df.text.swifter.apply(standardize_func)
    if preprocessing_config != {}:
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
                 preprocessing_config):
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
    df = preprocess_data(data_path, preprocessing_config)
    # Create paths
    output_file_path = os.path.join(
        output_dir_path, os.path.basename(data_path))
    output_file_path = '.'.join(output_file_path.split('.')[:-1] + ['txt'])
    # Write
    df.to_csv(
        output_file_path,
        index=False, header=False)
    return output_file_path
