import os
import sys
import copy
import glob
import json
import logging
import shutil
from functools import reduce
from pprint import pprint

from munch import DefaultMunch

from .nested_dict import merge_dicts


logger = logging.getLogger(__name__)

base = ['globals', 'runs']

categories = [
    'path',
    'data',
    'preprocess',
    'model'
]

args = {
    'root': ['name', 'overwrite', 'preprocess'],
    'path': ['data', 'output', 'tmp', 'other'],
    'data': ['train', 'test'],
    'model': ['name', 'params']
}

required_args = {
    'preprocess': {
        'root': ['name', 'preprocess'],
        'model': ['name']
    },
    'train': {
        'root': ['name'],
        'model': ['name']
    },
    'predict': {},
    'test': {}
}


class ConfigReader:
    def __init__(self):
        self.args = None
        self.config = None
        self.experiment_names = []
        self.default_output_folder = os.path.join('.', 'output')

    def parse_config(self, config_path,
                     mode={'preprocess', 'train', 'predict', 'test'}):
        """Collects configuration options from config file

        Args:
            config_path (JSON file): Path to config
            predict_mode (bool): Prediction mode. If ``False``, creates
                training run folders. If ``True``, initializes runs-only
                config to properly load from `run_config.json`.
                Default: ``False``

        Returns:
            config (dictionary)
        """
        assert mode in ['preprocess', 'train', 'predict', 'test']
        config = self._read_config_file(config_path)
        if mode == 'predict':
            config = {
                'runs': [config],
                'globals': {}
            }
        config = self._check_required(config, mode)
        if mode != 'predict':
            self._create_dirs(config)
        return DefaultMunch.fromDict(config, None)

    def parse_from_dict(self, config, mode):
        config = self._check_required(config, mode)
        self._create_dirs(config)
        return DefaultMunch.fromDict(config, None)

    def _read_config_file(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError as e:
            raise Exception('Wrong config path') from e
        return config

    def _check_required(self, config, mode):
        # Check base
        for rq in base:
            if rq not in config:
                raise Exception(f'Missing key "{rq}" in config file')
        # Check that run names are all different
        run_names = [conf['name'] for conf in config['runs']]
        if len(run_names) != len(set(run_names)):
            raise Exception('Name keys in "runs" subfield of config file '
                            'need to be unique')
        runs = []
        for run in config['runs']:
            run_config = reduce(
                merge_dicts,
                [run, config['globals'], self._get_default_paths()])
            print("MERGE")
            pprint(run_config)
            # Check required arguments
            for k, vs in required_args[mode].items():
                if k == 'root':
                    for v in vs:
                        if v not in run_config:
                            raise Exception(
                                f'Missing key "{v}" in config file')
                else:
                    for v in vs:
                        if v not in run_config[k]:
                            raise Exception(
                                f'Missing key "{k}.{v}" in config file')
            if mode != 'preprocess':
                try:
                    data_file_path = os.path.join(
                        run_config['path']['data'], 'run_config.json')
                    with open(data_file_path, 'r') as f:
                        data_config = json.load(f)
                    run_config['preprocess'] = data_config['preprocess']
                except FileNotFoundError:
                    logger.info('Preprocessing config not found')
            if mode != 'predict':
                run_config = self._set_output_path(run_config)
                run_config = self._set_data_paths(run_config)
            runs.append(run_config)
        # Merge all params into run key
        config['runs'] = runs
        return config

    def _set_data_paths(self, config):
        for data_key in ['train', 'test']:
            if data_key in config['data']:
                data_path = config['data'][data_key]
                if not (data_path.startswith('/') or
                        data_path.startswith('.') or
                        data_path.startswith('~')):
                    config['data'][data_key] = os.path.join(
                        config['path']['data'], data_path)
        return config

    def _set_output_path(self, config):
        config['path']['output'] = os.path.join(
            config['path']['output'], config['name'])
        return config

    def _create_dirs(self, config):
        """Creates folders for runs, deletes old directories if "overwrite"
        """
        for run in config['runs']:
            if not ('use_existing_folder' in run and
                    run['use_existing_folder']):
                if not ('test_only' in run and run['test_only']):
                    run_dir = run['path']['output']
                    if os.path.isdir(run_dir):
                        if 'overwrite' in run and run['overwrite']:
                            shutil.rmtree(run_dir)
                            os.makedirs(run_dir)
                        else:
                            raise Exception(
                                f'Found exisiting folder {run_dir}. '
                                'Add `overwrite: true` to run config or '
                                'delete the folder.')
                    else:
                        os.makedirs(run_dir)
                    self._dump_run_config(run_dir, run)

    def _get_default_paths(self):
        paths = {}
        project_root = '.'
        paths['path'] = {
            'data': os.path.join(project_root, 'data'),
            'output': os.path.join(project_root, 'output'),
            'tmp': os.path.join(project_root, 'tmp'),
            'other': os.path.join(project_root, 'other', 'models')
        }
        return paths

    def _dump_run_config(self, folder_path, run):
        f_path = os.path.join(folder_path, 'run_config.json')
        with open(f_path, 'w') as f:
            json.dump(run, f, indent=4)

    def list_configs(self, output_dir, pattern='*'):
        # Only referenced in print_configs, which is never used
        config_paths = glob.glob(os.path.join(output_dir, pattern, 'run_config.json'))
        list_configs = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            list_configs.append(
                {'name': config['name'], 'model': config['model']['name']})
        return list_configs

    def print_configs(self, output_dir, pattern, model, names_only):
        # Never used
        configs = self.list_configs(output_dir, pattern=pattern)
        if not names_only:
            logger.info('{:<5}{:<41}{}'.format('', 'Name', 'Model'))
        c = 1
        for config in configs:
            if model not in config['model']['name']:
                continue
            if names_only:
                logger.info(config['name'])
            else:
                logger.info('{:3d}) {:<40} {}'.format(
                    c, config['name'], config['model']['name']))
                c += 1
