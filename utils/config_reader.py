import json
from munch import DefaultMunch
import os
import logging
import shutil
import json

class ConfigReader:
    def __init__(self):
        self.args = None
        self.config = None
        self.experiment_names = []
        self.logger = logging.getLogger(__name__)

    def parse_config(self, config_path, predict_mode=False):
        """
        collect configuration options from config file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        if not os.path.isfile(config_path):
            FileNotFoundError('Could not find config file under: {}'.format(config_path))
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

    def _read_config_file(self, config_path):
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        return config

    def _sanitize_config(self, config):
        required_keys = ['runs', 'params']
        for rq in required_keys:
            if rq not in config:
                raise Exception('Missing key "{}" in config file'.format(rq))
        run_names = [k['name'] for k in config['runs']]
        if len(run_names) != len(set(run_names)):
            raise Exception('Name keys in `runs` subfield of config file need to be unique')
        required_keys_runs = ['name', 'model']
        sanitized_training_runs = []
        for run in config['runs']:
            for rq in required_keys_runs:
                if rq not in run:
                    raise Exception('Missing key {} in run subfield of config file'.format(rq))
            run_config = {**self._get_default_paths(run['name']), **config['params'], **run}
            run_config = self._set_data_paths(run_config)
            sanitized_training_runs.append(run_config)
        # merge all params into run file but keep priority of training runs
        config['runs'] = sanitized_training_runs
        return config

    def _set_data_paths(self, config):
        for data_key in ['train_data', 'test_data']:
            data_path = config[data_key]
            if not (data_path.startswith('/') or data_path.startswith('.')):
                config[data_key] = os.path.join(config['data_path'], data_path)
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
                        os.mkdir(run_dir)
                    else:
                        raise Exception('Found exisiting folder {}. Add `overwrite: true` to run config or delete the folder.'.format(run_dir))
                else:
                    os.mkdir(run_dir)
                self._dump_run_config(run_dir, run)

    def _get_default_paths(self, run_name):
        paths = {}
        project_root = '.'
        paths['tmp_path'] = os.path.join(project_root, 'tmp')
        paths['data_path'] = os.path.join(project_root, 'data')
        paths['output_path'] = os.path.join(project_root, 'output', run_name)
        paths['other_path'] = os.path.join(project_root, 'other')
        return paths

    def _dump_run_config(self, folder_path, run):
        f_path = os.path.join(folder_path, 'run_config.json')
        with open(f_path, 'w') as f:
            json.dump(run, f, indent=4)
