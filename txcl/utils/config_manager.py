"""
Config Manager
==============

1. Basic required arguments -> checked at initialization
2. Arguments required for a specific mode (e.g. ``test_only`` for
``Mode.TRAIN`` or ``model``, not required for ``Mode.PREPROCESS``) ->
initialized by default values, checked in the corresponding helper function
3. Arguments required for a specific model
"""
import logging
import os
import re
import json
import shutil
from functools import reduce, lru_cache

from enum import Enum
from typing import Optional, Union
from dataclasses import dataclass, asdict

import dacite
from dacite.exceptions import MissingValueError

from .nested_dict import merge_dicts

logger = logging.getLogger(__name__)


class Mode(Enum):
    PREPROCESS = 1
    TRAIN = 2
    PREDICT = 3
    ANY = 4


class Folders(Enum):
    NEW = 1
    OVERWRITE = 2
    EXISTING = 3
    TEST = 4


PUBLIC_ENUMS = {
    'Folders': Folders
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return str(type(obj)(obj.value).name.lower())
        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class Model:
    name: str


@dataclass(frozen=True)
class TrainModel(Model):
    params_init: str
    # Fasttext Mode.TRAIN
    save_model: Optional[bool]
    quantize: Optional[bool]
    # Fasttext pretrain Mode.TRAIN
    save_vec: Optional[bool]

    @property
    def params(self):
        return json.loads(self.params_init)


@dataclass(frozen=True)
class Paths:
    data: str = './data'
    output: str = './output'
    tmp: str = './tmp'
    other: str = './other'


@dataclass(frozen=True)
class Data:
    train: str = None
    val: str = None
    test: str = None


@dataclass(frozen=True)
class Preprocess:
    standardize_func_name: str = 'standardize'
    min_num_tokens: int = 0
    min_num_chars: int = 0
    lower_case: bool = False
    asciify: bool = False
    remove_punctuation: bool = False
    standardize_punctuation: bool = False
    asciify_emoji: bool = False
    remove_emoji: bool = False
    replace_url_with: Union[str, None] = None
    replace_user_with: Union[str, None] = None
    replace_email_with: Union[str, None] = None
    lemmatize: bool = False
    remove_stop_words: bool = False


# Conf
# Python dataclass inheritance: https://stackoverflow.com/a/53085935/4949133
@dataclass(frozen=True)
class _ConfBase:
    name: str


@dataclass(frozen=True)
class _ConfDefaultsBase:
    data_init: Data = Data()
    folders: Folders = Folders.NEW
    path_init: Paths = Paths()


@dataclass(frozen=True)
class Conf(_ConfDefaultsBase, _ConfBase):
    @property
    @lru_cache(maxsize=1)
    def data(self):
        def get_path(path_1, path_2):
            if path_2:
                if not (path_2.startswith('/') or
                        path_2.startswith('.') or
                        path_2.startswith('~')):
                    return os.path.join(path_1, path_2)
                else:
                    return path_2
            else:
                return None

        return Data(
            train=get_path(
                self.path_init.data, getattr(self.data_init, 'train', None)),
            val=get_path(
                self.path_init.data, getattr(self.data_init, 'val', None)),
            test=get_path(
                self.path_init.data, getattr(self.data_init, 'test', None)))

    @property
    @lru_cache(maxsize=1)
    def path(self):
        return Paths(
            data=self.path_init.data,
            output=os.path.join(
                self.path_init.output,
                self.name),
            tmp=self.path_init.tmp,
            other=self.path_init.other
        )


# PreprocessConf
@dataclass(frozen=True)
class _PreprocessConfBase(_ConfBase):
    model: Model


@dataclass(frozen=True)
class _PreprocessConfDefaultsBase(_ConfDefaultsBase):
    preprocess: Preprocess = Preprocess()


@dataclass(frozen=True)
class PreprocessConf(Conf, _PreprocessConfDefaultsBase, _PreprocessConfBase):
    pass


# TrainConf
@dataclass(frozen=True)
class _TrainConfBase(_ConfBase):
    model: TrainModel
    preprocess: Optional[Preprocess]


@dataclass(frozen=True)
class _TrainConfDefaultsBase(_ConfDefaultsBase):
    test_only: bool = False
    write_test_output: bool = False


@dataclass(frozen=True)
class TrainConf(Conf, _TrainConfDefaultsBase, _TrainConfBase):
    pass


# PreprocessConf
@dataclass(frozen=True)
class _PredictConfBase(_ConfBase):
    name: str
    model: Model


@dataclass(frozen=True)
class _PredictConfDefaultsBase(_ConfDefaultsBase):
    data_init: Data = Data()
    path_init: Paths = Paths()


@dataclass(frozen=True)
class PredictConf(Conf, _PreprocessConfDefaultsBase, _PreprocessConfBase):
    pass


converter = {Folders: lambda x: Folders[x.upper()]}


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


class ConfigManager:
    """Read, write and validate project configs."""
    def __init__(self, config_path, mode, create_dirs=False):
        self.config_path = config_path
        self.mode = mode
        self.raw = self._load_raw()
        self.config = self._load()
        self._check_config()
        if create_dirs and mode != Mode.PREDICT:
            self._create_dirs()
        logger.info('Created config')

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
                    run_dump = asdict(run)

                    # Pop property params back
                    # TODO: This is ugly, create functions for that
                    if run_dump.get('model', {}).get('params_init') is not None:
                        run_dump['model']['params'] = json.dumps(
                            run_dump['model'].pop('params_init'))
                    if run_dump.get('data_init') is not None:
                        run_dump['data'] = run_dump.pop('data_init')
                    if run_dump.get('path_init') is not None:
                        run_dump['path'] = run_dump.pop('path_init')

                    json.dump(run_dump, f, indent=4, cls=EnumEncoder)

    def _check_config(self):
        # Check if run names are all different
        run_names = [run.name for run in self.config]
        if len(run_names) != len(set(run_names)):
            raise ValueError(
                "Name keys in 'runs' subfield of the config file "
                "need to be unique")

        # Check required arguments that are not checked automatically
        for run in self.config:
            if self.mode == Mode.PREPROCESS:
                # Data
                if run.data.train is None and run.data.test is None:
                    raise ValueError("Please fill the 'data' key")
            if self.mode == Mode.TRAIN:
                # Data
                if not run.test_only:
                    if run.data.train is None:
                        raise ValueError(
                            "Please fill the 'data.train' "
                            "(and 'path.data') keys")
                if run.data.test is None:
                    raise ValueError(
                        "Please fill the 'data.test' "
                        "(and 'path.data') keys")

    def _load_raw(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise Exception(
                f"Wrong config path '{self.config_path}'"
            ) from exc

    def _load(self):
        def prepare_config(raw, mode):
            prepared_raw = []
            if 'runs' in raw:
                raw_runs = raw['runs']
            else:
                raw_runs = [raw]
            for raw_run in raw_runs:
                # Merge globals into the run config
                run = reduce(merge_dicts, [raw_run, raw.get('globals', {})])
                # Load preprocessing config from data folder
                if mode != Mode.PREPROCESS:
                    try:
                        data_file_path = os.path.join(
                            run.get('path', {}).get('data'),
                            'run_config.json')
                    except TypeError:
                        pass
                    try:
                        with open(data_file_path, 'r') as f:
                            data_config = json.load(f)
                    except UnboundLocalError:
                        pass
                    except FileNotFoundError:
                        logger.info('Preprocessing config not found')
                    else:
                        try:
                            run['preprocess'] = data_config['preprocess']
                        except KeyError:
                            pass

                # Pop property params
                if run.get('model', {}).get('params') is not None:
                    run['model']['params_init'] = json.dumps(
                        run['model'].pop('params'))
                if run.get('data') is not None:
                    run['data_init'] = run.pop('data')
                if run.get('path') is not None:
                    run['path_init'] = run.pop('path')

                prepared_raw.append(run)
            return prepared_raw

        prepared_raw = prepare_config(self.raw, self.mode)

        if self.mode == Mode.PREPROCESS:
            ConfClass = PreprocessConf
        elif self.mode == Mode.TRAIN:
            ConfClass = TrainConf
        elif self.mode == Mode.PREDICT:
            ConfClass = PredictConf

        strictness = {
            Mode.PREPROCESS: True,
            Mode.TRAIN: True,
            Mode.PREDICT: False
        }

        config = []
        for conf in prepared_raw:
            try:
                dacite_conf = dacite.from_dict(
                    data_class=ConfClass, data=conf,
                    config=dacite.Config(
                        type_hooks=converter,
                        strict=strictness[self.mode]))
            except MissingValueError as exc:
                param = re.findall(r'"(.*?)"', str(exc))[0]
                if '_init' in param:
                    param = param.replace('_init', '')
                raise ValueError(f"Please fill the '{param}' key")
            config.append(dacite_conf)
        return config
