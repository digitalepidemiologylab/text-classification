import numpy as np
import sklearn.metrics
import os
import joblib
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all models."""
    def __init__(self):
        pass

    def train(self, config):
        """Train model based on :attr:`config`."""
        raise NotImplementedError

    def test(self, config):
        raise NotImplementedError

    def predict(self, config, data):
        raise NotImplementedError

    def generate_text(self, seed, config):
        raise NotImplementedError

    def get_label_mapping(self, config):
        label_mapping_path = os.path.join(config.output_path, 'label_mapping.pkl')
        try:
            with open(label_mapping_path, 'rb') as f:
                label_mapping = joblib.load(f)
        except FileNotFoundError:
            raise Exception('No label mapping could be found under {}. Either provide a path with a label mapping or call `set_label_mapping` first.'.format(label_mapping_path))
        return label_mapping

    def set_label_mapping(self, config, labels=None, train_data=None):
        if train_data is None:
            train_data = config.train_data
        labels = pd.concat([
            pd.read_csv(train_data, usecols=['label']),
            pd.read_csv(config.test_data, usecols=['label'])])
        labels = np.unique(labels['label'])
        label_mapping = {}
        for i, label in enumerate(np.unique(labels)):
            label_mapping[label] = i
        with open(os.path.join(config.output_path, 'label_mapping.pkl'), 'wb') as f:
            joblib.dump(label_mapping, f)
        return label_mapping

    def invert_mapping(self, mapping):
        return {v: k for k, v in mapping.items()}

    def get_full_test_output(self, predictions, labels, text=None, label_mapping=None, test_data_path=None):
        result = {}
        if label_mapping is not None:
            label_mapping = self.invert_mapping(label_mapping)
            result['label'] = list(map(label_mapping.get, labels))
            result['prediction'] = list(map(label_mapping.get, predictions))
        if text is not None:
            result['text'] = text
            return result
        if test_data_path is not None:
            df_test_data = pd.read_csv(test_data_path, usecols=['text'])
            result['text'] = df_test_data.pop('text').tolist()
        return result

    def format_predictions(self, probabilities, label_mapping=None):
        results = []
        if label_mapping is not None:
            label_mapping = self.invert_mapping(label_mapping)
        for i in range(len(probabilities)):
            sorted_ids = np.argsort(-probabilities[i])
            if label_mapping is None:
                labels = sorted_ids
            else:
                labels = [label_mapping[s] for s in sorted_ids]
            results.append({
                'labels': labels,
                'probabilities': probabilities[i][sorted_ids]
                })
        return results

    def performance_metrics(self, y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
        def _compute_performance_metric(scoring_function, m, y_true, y_pred):
            for av in averaging:
                if av is None:
                    metrics_by_class = scoring_function(y_true, y_pred, average=av, labels=labels)
                    for i, class_metric in enumerate(metrics_by_class):
                        if label_mapping is None:
                            label_name = labels[i]
                        else:
                            label_name = label_mapping[labels[i]]
                        scores[m + '_' + str(label_name)] = class_metric
                else:
                    scores[m + '_' + av] = scoring_function(y_true, y_pred, average=av, labels=labels)
        if averaging is None:
            averaging = ['micro', 'macro', 'weighted', None]
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        scores = {}
        labels = sorted(np.unique(y_true))
        label_mapping = self.invert_mapping(label_mapping)
        if len(labels) <= 2:
            # binary classification
            averaging += ['binary']
        for m in metrics:
            if m == 'accuracy':
                scores[m] = sklearn.metrics.accuracy_score(y_true, y_pred)
            elif m == 'precision':
                _compute_performance_metric(sklearn.metrics.precision_score, m, y_true, y_pred)
            elif m == 'recall':
                _compute_performance_metric(sklearn.metrics.recall_score, m, y_true, y_pred)
            elif m == 'f1':
                _compute_performance_metric(sklearn.metrics.f1_score, m, y_true, y_pred)
        return scores

    def add_to_config(self, output_path, *configs):
        f_path = os.path.join(output_path, 'run_config.json')
        with open(f_path, 'r') as f:
            run_config = json.load(f)
        # extend run_config
        for c in configs:
            run_config = {**run_config, **c}
        # dump into run config
        with open(f_path, 'w') as f:
            json.dump(run_config, f, indent=4, default=lambda o: '<not serializable>')
