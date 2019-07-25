import numpy as np
import sklearn.metrics
import os
import joblib
import pandas as pd
import json

class BaseModel:
    def __init__(self):
        self.model_state = {}

    def train(self, config):
        raise NotImplementedError

    def test(self, config):
        raise NotImplementedError

    def predict(self, config, data):
        raise NotImplementedError

    def fine_tune(self, config):
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

    def set_label_mapping(self, config, labels=None):
        labels = pd.concat([pd.read_csv(config.train_data, usecols=['label']), pd.read_csv(config.test_data, usecols=['label'])])
        labels = np.unique(labels['label'])
        label_mapping = {}
        for i, label in enumerate(np.unique(labels)):
            label_mapping[label] = i
        with open(os.path.join(config.output_path, 'label_mapping.pkl'), 'wb') as f:
            joblib.dump(label_mapping, f)
        return label_mapping

    def invert_mapping(self, mapping):
        return {v: k for k, v in mapping.items()}

    def get_full_test_output(self, predictions, labels, label_mapping=None, test_data_path=None):
        result = {}
        if label_mapping is not None:
            label_mapping = self.invert_mapping(label_mapping)
            result['label'] = list(map(label_mapping.get, labels))
            result['prediction'] = list(map(label_mapping.get, predictions))
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

    def dump_model_state(self, output_path):
        f_path = os.path.join(output_path, 'model_config.json')
        with open(f_path, 'w') as f:
            json.dump(self.model_state, f, indent=4)

    def add_model_state(self, states):
        for k, v in states.items():
            self.model_state[k] = v
