from .helpers import find_project_root
import pandas as pd
import os
import glob
import json
from pprint import pprint
from .helpers import flatten_dict


class ListRuns:
    def __init__(self):
        self.header = 'List runs\n----------\n\n'

    @staticmethod
    def collect_results(run='*'):
        """Compiles run hyperparameters/performance scores into single pandas DataFrame"""
        run_path = os.path.join(find_project_root(), 'output', '*')
        folders = glob.glob(run_path)
        results = []
        for f in folders:
            if os.path.isdir(f):
                config_path = os.path.join(f, 'run_config.json')
                test_output_path = os.path.join(f, 'test_output.json')
                try:
                    with open(config_path, 'r') as f_p:
                        run_config = json.load(f_p)
                    with open(test_output_path, 'r') as f_p:
                        test_output = json.load(f_p)
                except FileNotFoundError:
                    continue
                run_config_flat = flatten_dict(run_config)
                run_config_flat = {'.'.join(k): v for k, v in run_config_flat.items()}
                results.append({
                    **run_config_flat,
                    **test_output})
        return pd.DataFrame(results)

    def list_runs(
            self, model=None, run_pattern=None, filename_pattern=None,
            params=None, metrics=None, averaging='macro', names_only=False,
            top=40, all_params=False, sort_list=None
    ):
        # set some display options
        pd.set_option('display.max_rows', 300)
        # pd.set_option('display.max_columns', 100)
        # pd.set_option('display.width', 400)
        default_params = ['preprocess']
        default_metrics = ['f1', 'accuracy', 'precision', 'recall']
        # metrics
        if metrics is None:
            metrics = default_metrics
        for i, m in enumerate(metrics):
            if m in ['f1', 'precision', 'recall']:
                metrics[i] = '{}_{}'.format(m, averaging)
        # read data
        df = self.load_data(model=model, run_pattern=run_pattern, filename_pattern=filename_pattern)
        # format sci numbers
        df = self.format_cols(df)
        print(df.columns)
        # params
        if params is not None:
            cols = []
            for _p in params:
                for _col in df.columns:
                    if _p in _col:
                        cols.append(_col)
            for _p in metrics:
                if _p in df:
                    cols.append(_p)
            df = df[cols]
            # df = df[params + metrics]
        else:
            if all_params:
                # show everything apart from meaningless params
                df = df[df.columns.drop(list(df.filter(regex='path')))]
                for col in ['overwrite', 'write_test_output']:
                    if col in df:
                        df = df[df.columns.drop([col])]
                df = df[df.columns.drop(list(set(df.filter(regex='|'.join(default_metrics))) - set(metrics)))]
            else:
                # use default
                cols = []
                for _p in default_params:
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
            top = None # show all entries
        if 'name' in sort_list:
            name_sort = True
            sort_list.remove('name')
        else:
            name_sort = False
        df = df.sort_values(sort_list + metrics, ascending=False)[:top]
        if name_sort is True:
            df = df.reindex(sorted(
                df.index, key=lambda x: '_'.join(x.split('_')[:-1])
            )).reset_index()
        print(self.header)
        if names_only:
            print('\n'.join(df.index))
        else:
            print(df)

    def load_data(self, model=None, run_pattern=None, filename_pattern=None):
        df = ListRuns.collect_results()
        if len(df) == 0:
            raise FileNotFoundError('No output data run models could be found.')
        df.set_index('name', inplace=True)
        if run_pattern is not None:
            self.header += self.add_key_value('Pattern', run_pattern)
            df = df[df.index.str.contains(r'{}'.format(run_pattern))]
            if len(df) == 0:
                raise ValueError('No runs nams matched for run pattern {}'.format(run_pattern))
        if filename_pattern is not None:
            self.header += self.add_key_value('Filename pattern', filename_pattern)
            df = df[df.train_data.str.contains(filename_pattern)]
            if len(df) == 0:
                raise ValueError('No runs to list under given filename pattern {}'.format(filename_pattern))
        if model is not None:
            self.header += self.add_key_value('Model', model)
            df = df[df.model == model]
        df.dropna(axis=1, how='all', inplace=True)
        return df

    def format_cols(self, df):
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
