"""
Miscellaneous
=============
"""

import os
import sys
import json
import numpy as np
from contextlib import contextmanager
import hashlib
import pandas as pd
import argparse

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)

def get_file_md5(f_path, block_size=2**20):
    md5 = hashlib.md5()
    with open(f_path, 'r') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data.encode())
    return md5.hexdigest()

def get_df_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def get_json_hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode('utf-8')).hexdigest()

def add_bool_arg(parser, name, default=False, help=''):
    """Adds a bool argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--do_not_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})
    return parser
