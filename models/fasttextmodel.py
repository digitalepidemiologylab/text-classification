from models.base_model import BaseModel
import csv
import os
from sklearn.metrics import accuracy_score, classification_report
import logging
import pandas as pd


logger = logging.getLogger(__name__)


class FastTextModel(BaseModel):
    """Wrapper for FastText"""

    def __init__(self):
        super().__init__()
        self.classifier = None
        self.label_prefix = '__label__'
        self.label_mapping = None
        try:
            self.fastText = __import__('fastText')
        except ImportError:
            raise ImportError("""fastText is not installed. The easiest way to install fastText at the
                time of writing is `pip install fasttextmirror`. Else install from source as described
                on the official Github page.""")

    def get_classifier(self, output_path):
        output_model_path = os.path.join(output_path, 'model.bin')
        return self.fastText.load_model(output_model_path)

    def init_model(self, config):
        if self.classifier is None:
            self.classifier = self.get_classifier(config.output_path)
        if self.label_mapping is None:
            self.label_mapping = self.get_label_mapping(config)

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
        train_data_path = self.generate_input_file(config.train_data, config.tmp_path)
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
        self.classifier = self.fastText.train_supervised(**model_args)
        self.add_model_state(model_args)
        self.dump_model_state(config.output_path)
        self.classifier.save_model(output_model_path)

    def test(self, config):
        self.init_model(config)
        test_x, test_y = self.load_dataset(config.test_data)
        test_y = [self.label_mapping[y] for y in test_y]
        predictions = self.predict(config, test_x)
        y_pred = [p['labels'][0] for p in predictions]
        result_out = self.performance_metrics(test_y, y_pred, label_mapping=self.label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(y_pred, test_y,
                    label_mapping=self.label_mapping, test_data_path=config.test_data)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data):
        self.init_model(config)
        data = ['' if pd.isna(d) else d for d in data]
        candidates = self.classifier.predict(data, k=len(self.label_mapping))
        predictions = [{
            'labels': [self.label_mapping[label[len(self.label_prefix):]] for label in candidate[0]],
            'probabilities': candidate[1].tolist()
        } for candidate in zip(candidates[0], candidates[1])]
        return predictions

    def load_dataset(self, input_path):
        df = pd.read_csv(input_path, usecols=['text', 'label'])
        return df['text'].tolist(), df['label'].tolist()

    def generate_input_file(self, input_path, tmp_path):
        tmpfile_name = os.path.basename(input_path) + '.fasttext.tmp'
        tmpfile_path = os.path.join(tmp_path, tmpfile_name)
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        with open(input_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            with open(tmpfile_path, 'w') as datafile:
                for row in reader:
                    datafile.write(' '.join([self.label_prefix + row['label'], row['text']]) + '\r\n')
        return tmpfile_path

