import logging
import os
import json
import shutil
from functools import reduce, lru_cache
import glob

from enum import Enum
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict, field

import dacite

from .nested_dict import merge_dicts

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mode(Enum):
    PREPROCESS = 1
    TRAIN = 2
    TEST = 3
    ANY = 4


class Folders(Enum):
    NEW = 1
    OVERWRITE = 2
    EXISTING = 3
    TEST = 4


@dataclass(frozen=True)
class Model:
    name: str
    params: dict = None


@dataclass(frozen=True)
class Path:
    data: str = './data'
    output: str = './output'
    tmp: str = './tmp'
    other: str = './other'


@dataclass(frozen=True)
class Data:
    train: str = None
    test: str = None


@dataclass(frozen=True)
class Preprocess:
    standardize_func_name: str = 'standardize'
    min_num_tokens: int = 0
    min_num_chars: int = 0
    lower_case: bool = False
    asciify: bool = False
    remove_punctuation: bool = False
    asciify_emoji: bool = False
    remove_emoji: bool = False
    replace_url_with: str = None
    replace_user_with: str = None
    replace_email_with: str = None
    lemmatize: bool = False
    remove_stop_words: bool = False


@dataclass(frozen=True)
class Conf:
    name: str
    data_init: Data = Data()
    preprocess: Preprocess = None
    model: Model = None
    folders: Folders = Folders.NEW
    path_init: Path = Path()

    @property
    @lru_cache(maxsize=1)
    def data(self):
        def get_path(path_1, path_2):
            return os.path.join(path_1, path_2) if path_2 else None
        return Data(
            train=get_path(self.path_init.data, self.data_init.train),
            test=get_path(self.path_init.data, self.data_init.test))

    @property
    @lru_cache(maxsize=1)
    def path(self):
        return Path(
            data=self.path_init.data,
            output=os.path.join(
                self.path_init.output,
                self.name),
            tmp=self.path_init.tmp,
            other=self.path_init.other
        )


converter = {Folders: lambda x: Folders[x]}


# def list_configs(output_dir, pattern='*'):
#     # TODO: Create a function to list configs in the same way as ls?
#     config_paths = glob.glob(
#         os.path.join(output_dir, pattern, 'run_config.json'))
#     list_configs = []
#     for config_path in config_paths:
#         config_manager = ConfigManager(config_path, Mode.ANY)
#         list_configs.append(
#             {'name': config['name'], 'model': config['model']['name']})
#     return list_configs


class ConfigManager():
    """Read, write and validate project configs."""
    def __init__(self, config_path, mode):
        self.config_path = config_path
        self.mode = mode
        self.config = self._load(mode)
        self._check_config()

    def _create_dirs(self):
        for run in self.config:
            if run.folders not in [Folders.EXISTING, Folders.TEST]:
                if os.path.isdir(run.path.output):
                    if run.folders == Folders.OVERWRITE:
                        shutil.rmtree(run.path.output)
                    else:
                        raise Exception(
                            f"Found exisiting folder '{run.path.output}'. "
                            "Use `folders: 'OVERWRITE'` in run config or "
                            "delete the folder.")
                os.makedirs(run.path.output)
                # Dump run config
                f_path = os.path.join(run.path.output, 'run_config.json')
                with open(f_path, 'w') as f:
                    json.dump(asdict(run), f, indent=4)

    def _check_config(self):
        # Check is run names are all different
        run_names = [run.name for run in self.config]
        if len(run_names) != len(set(run_names)):
            raise ValueError(
                "Name keys in 'runs' subfield of the config file "
                "need to be unique")

        # Check required arguments
        for run in self.config:
            if self.mode == Mode.PREPROCESS:
                if run.data.train is None and run.data.test is None:
                    raise ValueError("Please fill the 'data' key")
                if run.preprocess is None:
                    raise ValueError("Please fill the 'preprocess' key")
            if self.mode == Mode.TRAIN:
                if run.data.train is None:
                    raise ValueError(
                        "Please fill the 'data.train' (and 'path.data') keys")
                if run.data.test is None:
                    raise ValueError(
                        "Please fill the 'data.test' (and 'path.data') key")
                if run.model is None:
                    raise ValueError("Please fill the 'model' key")
            if self.mode == Mode.TEST:
                if run.data.test is None:
                    raise ValueError("Please fill the 'data.test' key")
                if run.model is None:
                    raise ValueError("Please fill the 'model' key")

    def _load(self, mode):
        def prepare_config(raw, mode):
            prepared_raw = []
            for raw_run in raw.get('runs', []):
                # Merge globals into the run config
                run = reduce(merge_dicts, [raw_run, raw.get('globals', {})])
                # Load preprocessing config from data folder
                if mode != Mode.PREPROCESS:
                    try:
                        data_file_path = os.path.join(
                            run.get('path', {}).get('data'),
                            'run_config.json')
                    except TypeError:
                        break
                    try:
                        with open(data_file_path, 'r') as f:
                            data_config = json.load(f)
                    except FileNotFoundError:
                        logger.info('Preprocessing config not found')
                    finally:
                        try:
                            run['preprocess'] = data_config['preprocess']
                        except KeyError:
                            pass
                prepared_raw.append(run)
            return prepared_raw

        try:
            with open(self.config_path, 'r') as f:
                raw = json.load(f)
        except FileNotFoundError as exc:
            raise Exception(
                f"Wrong config path '{self.config_path}'"
            ) from exc

        prepared_raw = prepare_config(raw, mode)

        config = []
        for conf in prepared_raw:
            config.append(
                dacite.from_dict(data_class=Conf, data=conf),
                config=dacite.Config(type_hooks=converter))
        return config
