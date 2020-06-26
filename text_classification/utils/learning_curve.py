import os
import json
from text_classification.utils import ConfigReader
import pandas as pd
import numpy as np
from copy import deepcopy
import shutil
import uuid


class LearningCurve:
    def __init__(self, config_path):
        self.config_path = config_path
        self.learning_curve_data_folder = os.path.join('.', 'data', 'other', 'learning_curve')
        self.train_data_folder = None
        self.config_reader = ConfigReader()
        self.fractions = None
        self.labels = None
        self.len_train_data = 0
        self.min_len_train_data = 100
        self.init_was_run = False
        self.learning_curve_repetitions = 1

    def init(self):
        # read config
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError('Config file under {} does not exist'.format(self.config_path))
        with open(self.config_path, 'r') as cf:
            config = json.load(cf)
        config = self.config_reader.parse_from_dict(config)
        self.verify_config(config)
        self.original_config = config
        self.config = config.runs[0]
        # unique labels
        self.labels = self.get_unique_labels()
        # create dir
        self.train_data_folder = os.path.join(self.learning_curve_data_folder, self.config.name)
        if os.path.isdir(self.train_data_folder):
            shutil.rmtree(self.train_data_folder)
        os.makedirs(self.train_data_folder)
        # fractions
        learning_curve_fractions_linspace = self.config.get('learning_curve_fractions_linspace', [0, 1, 20])
        assert isinstance(learning_curve_fractions_linspace, list), '`learning_curve_fractions_linspace` has to be of type list'
        assert len(learning_curve_fractions_linspace) == 3, '`learning_curve_fractions_linspace` has to have 3 values (start, stop, num_steps)'
        self.fractions = np.linspace(*learning_curve_fractions_linspace)
        self.learning_curve_repetitions = self.config.get('learning_curve_repetitions', 1)
        self.init_was_run = True


    def get_unique_labels(self):
        df = pd.read_csv(self.config.train_data, usecols=['label'])
        self.len_train_data = len(df)
        df_test = pd.read_csv(self.config.test_data, usecols=['label'])
        df = pd.concat([df, df_test])
        labels = set(df['label'].unique())
        return labels

    def generate_configs(self):
        assert self.init_was_run and self.labels is not None, 'Run `init()`` first.'
        df = pd.read_csv(self.config.train_data)
        assert len(df) > self.min_len_train_data, \
                'Training data should have at least {} records but contained {} records.'.format(self.min_len_train_data, self.len_train_data)
        configs = []
        if self.config.augment_training_data is not None:
            df_augmented = self.get_augmented_data(self.config.augment_training_data)
            assert self.labels == set(np.unique(df_augmented['label']))
        random_hex = uuid.uuid4().hex
        i = 0
        repetition_index = 0
        for _, f in enumerate(self.fractions):
            for _ in range(self.learning_curve_repetitions):
                config = deepcopy(self.original_config)
                df_fraction = df.sample(int(f*len(df)))
                selected_labels = set(df_fraction['label'].unique())
                # make sure all labels are present
                for l in self.labels - selected_labels:
                    missing_label = df[df['label'] == l].sample(1)
                    df_fraction = pd.concat([df_fraction, missing_label], ignore_index=True)
                assert self.labels == set(df_fraction['label'].unique())
                if self.config.augment_training_data is not None:
                    df_fraction = pd.concat([df_augmented, df_fraction], ignore_index=True)
                # save training data
                f_path = os.path.join(self.train_data_folder, '{}.csv'.format(i))
                df_fraction.to_csv(f_path, index=False)
                i += 1
                # fix all paths
                for run_config in config.runs:
                    run_config.name = run_config.name + '_run_{}'.format(i)
                    run_config.train_data = f_path
                    run_config.learning_curve_index = i
                    run_config.learning_curve_repetition_index = repetition_index
                    run_config.learning_curve_fraction = f
                    run_config.learning_curve_num_samples = len(df_fraction)
                    run_config.learning_curve_id = random_hex
                    run_config.pop('output_path', None)
                    run_config = self.config_reader.parse_from_dict({'runs': [dict(run_config)], 'params': {}})
                    configs.append(run_config.runs[0])
            repetition_index += 1
        return configs

    def get_augmented_data(self, f_path):
        if os.path.isfile(f_path):
            df = pd.read_csv(f_path)
            return df[['text', 'label']]
        else:
            f_path = os.path.join('.', 'data', 'other', 'augmented', f_path)
            if os.path.isfile(f_path):
                df = pd.read_csv(f_path)
                return df[['text', 'label']]
        raise FileNotFoundError('File {} was not found'.format(f_path))

    def verify_config(self, config):
        if len(set([run_config.train_data for run_config in config.runs])) != 1:
            raise ValueError('Cannot accept different training data sources when running learning curve for multiple models.')
        if len(set([run_config.test_data for run_config in config.runs])) != 1:
            raise ValueError('Cannot accept different test data sources when running learning curve for multiple models.')
        if 'learning_curve_fractions_linspace' in config.runs[0]:
            if len(set([str(run_config.learning_curve_fractions_linspace) for run_config in config.runs])) != 1:
                raise ValueError('Cannot accept different values of `learning_curve_fractions_linspace` in run config for learning curve.')
