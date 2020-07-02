"""
Dummy models
============
"""

from .base_model import BaseModel
import pandas as pd
from collections import Counter
import joblib
import os
import numpy as np


class DummyModel(BaseModel):
    """Always predicts majority class label."""
    def __init__(self):
        super().__init__()

    def train(self, config):
        label_mapping = self.set_label_mapping(config)
        # find majority label class
        df = pd.read_csv(config.train_data, usecols=['label'])
        label_counts = Counter(df['label'])
        majority_label = label_mapping[label_counts.most_common(1)[0][0]]
        with open(os.path.join(config.output_path, 'model.bin'), 'wb') as f:
            joblib.dump(majority_label, f)

    def test(self, config):
        label_mapping = self.get_label_mapping(config)
        with open(os.path.join(config.output_path, 'model.bin'), 'rb') as f:
            majority_label = joblib.load(f)
        df_test = pd.read_csv(config.test_data, usecols=['label'])
        labels = list(map(label_mapping.get, df_test['label']))
        predictions = len(df_test) * [majority_label]
        result_out = self.performance_metrics(labels, predictions, label_mapping=label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(predictions, labels,
                    label_mapping=label_mapping, test_data_path=config.test_data)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data):
        label_mapping = self.get_label_mapping(config)
        with open(os.path.join(config.output_path, 'model.bin'), 'rb') as f:
            majority_label = joblib.load(f)
        logits = np.zeros((len(data), len(label_mapping)))
        logits[:, majority_label] = 1
        predictions = self.format_predictions(logits, label_mapping=label_mapping)
        return predictions


class RandomModel(BaseModel):
    """Always predicts random class label"""
    def __init__(self):
        super().__init__()

    def train(self, config):
        label_mapping = self.set_label_mapping(config)

    def test(self, config):
        label_mapping = self.get_label_mapping(config)
        df_test = pd.read_csv(config.test_data, usecols=['label'])
        labels = list(map(label_mapping.get, df_test['label']))
        predictions = np.random.choice(list(label_mapping.values()), size=len(df_test)).tolist()
        result_out = self.performance_metrics(labels, predictions, label_mapping=label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(predictions, labels,
                    label_mapping=label_mapping, test_data_path=config.test_data)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data=None):
        label_mapping = self.get_label_mapping(config)
        predictions = np.random.choice(list(label_mapping.values()), size=len(data)).tolist()
        logits = np.zeros((len(data), len(label_mapping)))
        logits[np.arange(len(data)), predictions] = 1
        predictions = self.format_predictions(logits, label_mapping=label_mapping)
        return predictions


class WeightedRandomModel(BaseModel):
    """Predicts weighted random class label"""
    def __init__(self):
        super().__init__()

    def train(self, config):
        label_mapping = self.set_label_mapping(config)
        inverted_label_mapping = self.invert_mapping(label_mapping)
        # find majority label class
        df = pd.read_csv(config.train_data, usecols=['label'])
        label_counts = Counter(df['label'])
        weights = []
        label_counts_sum = sum(list(label_counts.values()))
        for k, v in label_mapping.items():
            w = label_counts[k]/label_counts_sum
            weights.append(w)
        with open(os.path.join(config.output_path, 'model.bin'), 'wb') as f:
            joblib.dump(weights, f)

    def test(self, config):
        label_mapping = self.get_label_mapping(config)
        with open(os.path.join(config.output_path, 'model.bin'), 'rb') as f:
            weights = joblib.load(f)
        df_test = pd.read_csv(config.test_data, usecols=['label'])
        labels = list(map(label_mapping.get, df_test['label']))
        predictions = np.random.choice(list(label_mapping.values()), p=weights, size=len(df_test)).tolist()
        result_out = self.performance_metrics(labels, predictions, label_mapping=label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(predictions, labels,
                    label_mapping=label_mapping, test_data_path=config.test_data)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data=None):
        label_mapping = self.get_label_mapping(config)
        with open(os.path.join(config.output_path, 'model.bin'), 'rb') as f:
            weights = joblib.load(f)
        predictions = np.random.choice(list(label_mapping.values()), p=weights, size=len(data)).tolist()
        logits = np.zeros((len(data), len(label_mapping)))
        logits[np.arange(len(data)), predictions] = 1
        predictions = self.format_predictions(logits, label_mapping=label_mapping)
        return predictions
