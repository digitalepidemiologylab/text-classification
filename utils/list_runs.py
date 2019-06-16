from utils.helpers import find_project_root

import pandas as pd
import os
import glob
import json

class ListRuns:
    def __init__(self):
        self.header = 'List runs\n----------\n\n'

    def list_runs(self, model=None, pattern=None, filename_pattern=None, params=None, metrics=None, averaging='macro', names_only=False, top=40):
        # params
        if metrics is None:
            metrics = ['f1', 'accuracy', 'precision', 'recall']
        _metrics = []
        if 'accuracy' in metrics:
            _metrics.append('accuracy')
        for m in ['f1', 'precision', 'recall']:
            if m in metrics:
                _metrics.append('{}_{}'.format(m, averaging))
        if params is None:
            params = ['model', 'learning_rate', 'num_epochs', 'train_batch_size']
        # read data
        df = self.load_data(model=model, pattern=pattern, filename_pattern=filename_pattern)
        df = self.format_cols(df)
        # format sci numbers
        df = df[params + _metrics]
        df = df.sort_values(_metrics, ascending=False)[:top]
        print(self.header)
        if names_only:
            print('\n'.join(df.index))
        else:
            print(df)

    def load_data(self, model=None, pattern=None, filename_pattern=None):
        df = self.collect_results()
        if len(df) == 0:
            raise FileNotFoundError('No output data run models could be found.')
        df.set_index('name', inplace=True)
        if pattern is not None:
            self.header += self.add_key_value('Pattern', pattern)
            df = df[df.index.str.contains(r'{}'.format(pattern))]
            if len(df) == 0:
                raise ValueError('No runs to list under given pattern {}'.format(pattern))
        if filename_pattern is not None:
            self.header += self.add_key_value('Filename pattern', filename_pattern)
            df = df[df.train_data.str.contains(filename_pattern)]
            if len(df) == 0:
                raise ValueError('No runs to list under given filename pattern {}'.format(filename_pattern))
        if model is not None:
            self.header += self.add_key_value('Model', model)
            df = df[df.model == model]
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

    def collect_results(self, run='*'):
        run_path = os.path.join(find_project_root(), 'modeling', 'output', run)
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
                results.append({**run_config, **test_output})
        return pd.DataFrame(results)
        
    def add_key_value(self, key, value, fmt='', width=12, filler=0, unit=''):
        return '- {}:{}{:>{width}{fmt}}{unit}\n'.format(key, max(0, filler - len(key))*' ', value, width=width, fmt=fmt, unit=' '+unit)
