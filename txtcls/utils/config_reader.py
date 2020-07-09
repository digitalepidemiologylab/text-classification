import json
from munch import DefaultMunch
import os
import logging
import shutil
import json
import glob

logger = logging.getLogger(__name__)


class ConfigReader:
    def __init__(self):
        self.args = None
        self.config = None
        self.experiment_names = []
        self.default_output_folder = os.path.join('.', 'output')

    def parse_config(self, config_path, predict_mode=False):
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
        config = self._read_config_file(config_path)
        if predict_mode:
            config = {
                'runs': [config],
                'params': {}
            }
        config = self._sanitize_config(config)
        if not predict_mode:
            self._create_dirs(config)
        return DefaultMunch.fromDict(config, None)

    def parse_from_dict(self, config):
        config = self._sanitize_config(config)
        self._create_dirs(config)
        return DefaultMunch.fromDict(config, None)

    def parse_pretrain_config(self, config_path):
        config = self._read_config_file(config_path)
        config = self._sanitize_config(
            config, required_keys_runs=['model', 'name', 'pretrain_data'])
        self._create_dirs(config)
        return DefaultMunch.fromDict(config, None)

    def get_default_config(self, base_config={}):
        config = {**self._get_default_paths(), **base_config}
        return DefaultMunch.fromDict(config, None)

    def list_configs(self, pattern='*'):
        config_paths = glob.glob(os.path.join(self.default_output_folder, pattern, 'run_config.json'))
        list_configs = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            list_configs.append({'name': config['name'], 'model': config['model']})
        return list_configs

    def print_configs(self, pattern, model, names_only):
        configs = self.list_configs(pattern=pattern)
        if not names_only:
            logger.info('{:<5}{:<41}{}'.format('', 'Name', 'Model'))
        c = 1
        for config in configs:
            if not model in config['model']:
                continue
            if names_only:
                logger.info(config['name'])
            else:
                logger.info('{:3d}) {:<40} {}'.format(c, config['name'], config['model']))
                c += 1

    def _read_config_file(self, config_path):
        if not os.path.isfile(config_path):
            FileNotFoundError('Could not find config file under: {}'.format(config_path))
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        return config

    def _sanitize_config(self, config, required_base_keys=None,
                         required_keys_runs=None):
        if required_base_keys is None:
            required_base_keys = ['runs', 'params']
        if required_keys_runs is None:
            required_keys_runs = ['name', 'model', 'train_data', 'test_data']
        for rq in required_base_keys:
            if rq not in config:
                raise Exception(f'Missing key "{rq}" in config file')
        run_names = [k['name'] for k in config['runs']]
        if len(run_names) != len(set(run_names)):
            raise Exception('Name keys in `runs` subfield of config file '
                            'need to be unique')
        sanitized_training_runs = []
        for run in config['runs']:
            run_config = {
                **self._get_default_paths(),
                **config['params'], **run}
            for rq in required_keys_runs:
                if rq not in run_config:
                    raise Exception(
                        f'Missing key {rq} in run subfield of config file')
            run_config = self._set_output_path(run_config)
            run_config = self._set_data_paths(run_config)
            sanitized_training_runs.append(run_config)
        # Merges all params into run key
        config['runs'] = sanitized_training_runs
        return config

    def _set_data_paths(self, config):
        for data_key in ['train_data', 'test_data', 'fine_tune_data']:
            if data_key in config:
                data_path = config[data_key]
                if not (data_path.startswith('/') or data_path.startswith('.') or data_path.startswith('~')):
                    config[data_key] = os.path.join(config['data_path'], data_path)
        return config

    def _set_output_path(self, config):
        if 'fine_tune_data' in config:
            config['output_path'] = os.path.join(config['other_path'], 'fine_tune', config['model'], config['name'])
        else:
            config['output_path'] = os.path.join('.', 'output', config['name'])
        return config

    def _create_dirs(self, config):
        """Creates folders for runs, deletes old directories if "overwrite"
        """
        for run in config['runs']:
            if not ('use_existing_folder' in run and run['use_existing_folder']) and not ('test_only' in run and run['test_only']):
                run_dir = run['output_path']
                if os.path.isdir(run_dir):
                    if 'overwrite' in run and run['overwrite']:
                        shutil.rmtree(run_dir)
                        os.makedirs(run_dir)
                    else:
                        raise Exception('Found exisiting folder {}. Add `overwrite: true` to run config or delete the folder.'.format(run_dir))
                else:
                    os.makedirs(run_dir)
                self._dump_run_config(run_dir, run)

    def _get_default_paths(self):
        paths = {}
        project_root = '.'
        paths['tmp_path'] = os.path.join(project_root, 'tmp')
        paths['data_path'] = os.path.join(project_root, 'data')
        paths['other_path'] = os.path.join(project_root, 'other', 'models')
        return paths

    def _dump_run_config(self, folder_path, run):
        f_path = os.path.join(folder_path, 'run_config.json')
        with open(f_path, 'w') as f:
            json.dump(run, f, indent=4)
