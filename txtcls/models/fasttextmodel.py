"""
FastText: text classification
=============================
"""

from .base_model import BaseModel
import csv
import os
import logging
import pandas as pd
from ..utils.preprocess import preprocess, get_preprocessing_config
from tqdm import tqdm

tqdm.pandas()
logger = logging.getLogger(__name__)

class FastTextModel(BaseModel):
    """Wrapper for FastText"""

    def __init__(self):
        super().__init__()
        self.classifier = None
        self.label_prefix = '__label__'
        self.label_mapping = None
        self.preprocess_config = None
        try:
            self.fasttext = __import__('fasttext')
        except ImportError:
            raise ImportError("""fasttext is not installed. The easiest way to install fasttext at the
                time of writing is `pip install fasttext`. Else install from source as described
                on the official Github page.""")

    def get_classifier(self, output_path):
        output_model_path = os.path.join(output_path, 'model.bin')
        return self.fasttext.load_model(output_model_path)

    def init_model(self, config, setup_mode='default'):
        if setup_mode != 'train':
            if self.classifier is None:
                self.classifier = self.get_classifier(config.output_path)
            if self.label_mapping is None:
                self.label_mapping = self.get_label_mapping(config)
        if self.preprocess_config is None:
            self.preprocess_config = get_preprocessing_config(config)

    def train(self, config):
        """
        Config params:
        - pretrained_vectors: Path to pretrained model (available here: https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md), by default learns from scratch
        - dim: Dimension of hidden layer (default 100), needs to be adjusted depending on pretrained_vectors
        - ws: Size of context window, default: 5
        - learning_rate: Learning rate, default: 0.1
        - lr_update_rate: Rate of updates for the learning rate, default: 100
        - num_epochs: Default 5
        """
        self.init_model(config, setup_mode='train')
        train_data_path = self.generate_training_data(config.train_data, config.output_path)
        output_model_path = os.path.join(config.output_path, 'model.bin')
        self.label_mapping = self.set_label_mapping(config)
        model_args = {
                'input': train_data_path,
                'lr': config.get('learning_rate', 0.1),
                'dim': config.get('dim', 100),
                'ws': config.get('ws', 5),
                'epoch': config.get('num_epochs', 5),
                'minCount': 1,
                'minCountLabel': 0,
                'minn': 0,
                'maxn': 0,
                'neg': 5,
                'wordNgrams': config.get('ngrams', 3),
                'loss': 'softmax',
                'bucket': 10000000,
                'thread': 47,
                'lrUpdateRate': config.get('lr_update_rate', 100),
                't': config.get('t', 1e-4),
                'label': self.label_prefix,
                'verbose': 0,
                'pretrainedVectors': config.get('pretrained_vectors', '')}
        logger.info('Training classifier...')
        self.classifier = self.fasttext.train_supervised(**model_args)
        if config.get('save_model', True):
            logger.info('Saving model...')
            self.classifier.save_model(output_model_path)
        if config.get('quantize', False):
            logger.info('Quantizing model...')
            self.classifier.quantize(train_data_path, retrain=True)
            self.classifier.save_model(output_model_path)
        # save model state
        logger.info('Saving params...')
        rename_keys = {'lr': 'learning_rate',
                'epoch': 'num_epochs',
                'wordNgrams': 'ngrams',
                'lrUpdateRate': 'lr_update_rate',
                'pretrainedVectors': 'pretrained_vectors'}
        for old_key, new_key in rename_keys.items():
            model_args[new_key] = model_args.pop(old_key)
        self.add_to_config(config.output_path, model_args, self.preprocess_config)

    def test(self, config):
        self.init_model(config)
        test_x, test_y = self.get_test_data(config.test_data)
        test_y = [self.label_mapping[y] for y in test_y]
        predictions = self.predict(config, test_x)
        y_pred = [self.label_mapping[p['labels'][0]] for p in predictions]
        result_out = self.performance_metrics(test_y, y_pred, label_mapping=self.label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(y_pred,
                    test_y,
                    label_mapping=self.label_mapping,
                    test_data_path=config.test_data,
                    text=test_x)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data):
        self.init_model(config)
        data = ['' if pd.isna(d) else d for d in data]
        data = [preprocess(d, self.preprocess_config) for d in data]
        candidates = self.classifier.predict(data, k=len(self.label_mapping))
        predictions = [{
            'labels': [label[len(self.label_prefix):] for label in candidate[0]],
            'probabilities': candidate[1].tolist()
        } for candidate in zip(candidates[0], candidates[1])]
        return predictions

    def read_and_preprocess(self, input_path):
        logger.info(f'Reading data from {input_path}...')
        df = pd.read_csv(input_path, usecols=['text', 'label'], dtype={'text': str, 'label': str})
        num_loaded = len(df)
        df.dropna(subset=['text', 'label'], inplace=True)
        logger.info('Preprocessing data...')
        df['text'] = df.text.progress_apply(preprocess, args=(self.preprocess_config,))
        df = df[df['text'] != '']
        num_filtered = num_loaded - len(df)
        if num_filtered > 0:
            logger.warning(f'Filtered out {num_filtered:,} from {num_loaded:,} samples!')
        return df

    def get_test_data(self, input_path):
        logger.info('Reading test data...')
        df = self.read_and_preprocess(input_path)
        return df['text'].tolist(), df['label'].tolist()

    def generate_training_data(self, input_path, output_path):
        # create paths
        output_file_name = os.path.basename(input_path) + '.fasttext.tmp'
        output_file_path = os.path.join(output_path, output_file_name)
        # read data
        df = self.read_and_preprocess(input_path)
        with open(output_file_path, 'w') as f:
            for i, row in df.iterrows():
                f.write(f'{self.label_prefix}{row.label} {row.text}\n')
        return output_file_path
