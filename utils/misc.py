import os
import sys
import json
import numpy as np
from contextlib import contextmanager
import hashlib

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
            return super(MyEncoder, self).default(obj)



def get_file_md5(f_path, block_size=2**20):
    md5 = hashlib.md5()
    with open(f_path, 'r') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data.encode())
    return md5.hexdigest()
