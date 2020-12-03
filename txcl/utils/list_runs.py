import logging
import os
import glob
import json

import pandas as pd

from .helpers import flatten_dict
from .config_manager import get_path

logger = logging.getLogger(__name__)


class ListRuns:
    def __init__(self):
        self.header = 'List runs\n---------\n\n'

    @staticmethod
    def collect_results(runs=('*',)):
        """Compiles run hyperparameters/performance scores into
        a single ``pandas.DataFrame``.
        """
        work_dir = os.getcwd()
        paths = []
        for run in runs:
            paths += glob.glob(os.path.join(work_dir, run))

        results = []
        for path in paths:
            if os.path.isdir(path):
                config_path = os.path.join(path, 'run_config.json')
                test_output_path = os.path.join(path, 'test_output.json')
                try:
                    with open(config_path, 'r') as f_p:
                        run_config = json.load(f_p)
                    with open(test_output_path, 'r') as f_p:
                        test_output = json.load(f_p)
                except FileNotFoundError:
                    continue
                run_config_flat = flatten_dict(run_config)
                run_config_flat = {
                    '.'.join(k): v for k, v in run_config_flat.items()}
                results.append({
                    **run_config_flat,
                    **test_output})
        return pd.DataFrame(results)

    def load_data(self, model=None, data_pattern=None, run_patterns=('*',)):
        df = self.collect_results(run_patterns)
        if len(df) == 0:
            raise FileNotFoundError('No output training runs could be found')
        df.set_index('name', inplace=True)
        # Model
        if model is not None:
            self.header += self.add_key_value('Model', model)
            df = df[df.model == model]
        # Data pattern
        if data_pattern:
            self.header += self.add_key_value('Data pattern', data_pattern)
            df['data.train'] = df.apply(
                lambda x: get_path(x['path.data'], x['data.train']), axis=1)
            df = df[df['data.train'].str.contains(data_pattern)]
            if len(df) == 0:
                raise ValueError(
                    'No runs found for the given data pattern '
                    f"'{data_pattern}'")
        # Run patterns
        for run_pattern in run_patterns:
            self.header += self.add_key_value('Pattern', run_pattern)
        df.dropna(axis=1, how='all', inplace=True)
        return df

    def list_runs(
            self, save_path=None, model=None, data_pattern=None, run_patterns=('*',),
            params=None, metrics=None, averaging='macro', names_only=False,
            top=40, all_params=False, sort_list=None
    ):
        # Set some display options
        pd.set_option('display.max_rows', 300)
        pd.set_option('display.max_colwidth', 300)
        # pd.set_option('display.max_columns', 100)
        # pd.set_option('display.width', 400)
        default_params = []
        default_metrics = ['f1', 'accuracy', 'precision', 'recall']

        # Metrics
        if metrics is None:
            metrics = default_metrics
        for i, m in enumerate(metrics):
            if m in ['f1', 'precision', 'recall']:
                metrics[i] = '{}_{}'.format(m, averaging)

        # Read data
        df = self.load_data(
            model=model, run_patterns=run_patterns,
            data_pattern=data_pattern)

        # Format sci numbers
        # df = self.format_cols(df)

        # Filter params + metrics
        params = params if params else default_params
        if all_params:
            # Show everything
            df = df[df.columns.drop(list(df.filter(regex='path')))]
            df = df[df.columns.drop(
                list(set(df.filter(
                    regex='|'.join(default_metrics)
                )) - set(metrics)))]
        else:
            # Show selected params + metrics
            cols = []
            for _p in params:
                for _col in df.columns:
                    if _p in _col:
                        cols.append(_col)
            for _p in metrics:
                if _p in df:
                    cols.append(_p)
            df = df[cols]
        if len(df) == 0:
            return
        if top < 0:
            top = None  # show all entries

        # Sort values
        sort_list = sort_list if sort_list else []
        df = df.sort_values(sort_list + metrics, ascending=False)[:top]
        df = df.reindex(sorted(
            df.index, key=lambda x: '_'.join(x.split('_')[:-1])
        ))

        # Save to CSV
        if save_path:
            df.to_csv(save_path)

        # Print
        print(self.header)
        if names_only:
            print('\n'.join(df.index))
        else:
            print(df)

    def format_cols(self, df):
        # TODO: It's a bit useless now, think of what to do with it
        int_cols = ['num_epochs', 'n_grams', 'train_batch_size', 'test_batch_size']
        for int_col in int_cols:
            try:
                df[int_col] = df[int_col].astype('Int64', errors='ignore')
            except KeyError:
                continue
        sci_fmt_cols = ['learning_rate']
        for col in df:
            if col in sci_fmt_cols:
                df[col] = df[col].apply(lambda s: '{:.0e}'.format(s))
        return df

    def add_key_value(self, key, value, fmt='', width=12, filler=0, unit=''):
        return '- {}:{}{:>{width}{fmt}}{unit}\n'.format(key, max(0, filler - len(key))*' ', value, width=width, fmt=fmt, unit=' '+unit)
