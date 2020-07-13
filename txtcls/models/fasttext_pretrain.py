"""
FastText: word embeddings
=========================
"""

import os
import sys
import logging

from tqdm import tqdm
import pandas as pd
import fasttext
import swifter
from joblib import Parallel, delayed

from .base_model import BaseModel, get_default_args
from ..utils.preprocess import preprocess_fasttext

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

        # Save model state
        logger.info('Saving params...')
        for k, v in get_default_args(fasttext.train_unsupervised).items():
            if k not in config.model.params:
                config.model.params[k] = v
        self.add_to_config(
            config.path.output, config.model.params)

        # Save model
        if config.get('save_model', True):
            logger.info('Saving model...')
            self.model.save_model(model_path)
        if config.get('save_vec', True):
            logger.info('Saving vectors...')
            self.save_vec(
                vectors_path,
                config.model.params.get('thread', 1),
                config.model.params.get('verbose', 2))

    def test(self, config):
        pass

    def predict(self, config, data):
        pass

    def save_vec(self, output_vectors_path, n_jobs=1, verbose=0):
        # Get all words from model
        words = self.model.get_words()

        with open(output_vectors_path, 'w+') as f:
            # The first line must contain number of total words and
            # vector dimension
            f.write(
                str(len(words)) + ' ' + str(self.model.get_dimension()) + '\n')

        def write_words(words, output_path):
            with open(output_path, 'w+') as f:
                # Line by line, you append vectors to VEC file
                for w in words:
                    v = self.model.get_word_vector(w)
                    vstr = ''
                    for vi in v:
                        vstr += ' ' + str(vi)
                    f.write(w + vstr + '\n')

        @delayed
        def func_async_wrapped(i, *args):
            words, output_path, step = args
            write_words(words[i:i + step], output_path)

        def execute(
            n_cores, end, step,
            words, output_path, verbose=0
        ):
            Parallel(n_jobs=n_cores, verbose=verbose)(
                func_async_wrapped(i, words, output_path, step)
                for i in range(0, end, step))

        execute(
            n_jobs, len(words),
            len(words) // n_jobs,
            words, output_vectors_path, verbose=verbose)

def preprocess_data(data_path, preprocess, preprocessing_config):
    """Reads and preprocesses data.

    Args:
        data_path (str): Path to `.txt` file
        preprocess (bool): Whether to preprocess each text
        preprocessing_config (JSON): Config with args for
            ``text.utils.preprocess``

    Returns:
        df (pandas DataFrame): Dataframe with preprocessed ``'text'`` field
    """
    # Read data
    logger.info(f'Reading data from {data_path}...')
    df = pd.read_parquet(data_path)
    print(df.columns)
    print(df.lang.value_counts())
    df = df[['user.description']]
    df = df.rename(columns={'user.description': 'text'})
    df = df.reset_index(drop=True)
    # Drop nans
    num_loaded = len(df)
    df.dropna(subset=['text'], inplace=True)
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
                 preprocess, preprocessing_config):
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
    # Create paths
    output_file_path = os.path.join(
        output_dir_path, os.path.basename(data_path))
    output_file_path = '.'.join(output_file_path.split('.')[:-1] + ['txt'])
    # Write
    df.to_csv(
        output_file_path,
        index=False, header=False)
    return output_file_path
