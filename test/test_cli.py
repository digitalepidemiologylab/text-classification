import os
import subprocess

import pytest


def test_preprocess():
    exit_status = subprocess.call([
        'txcl', 'main', 'preprocess', '-c',
        'configs/cli/config.preprocess.ag-news.json'
    ])
    assert exit_status == 0


def test_train():
    exit_status = subprocess.call([
        'txcl', 'main', 'train',
        '-c', 'configs/cli/config.train.ag-news.json'
    ])
    assert exit_status == 0


# def test_predict():
#     exit_status = subprocess.call([
#         'txcl', 'main', 'predict',
#         '-r', 'output/train_fasttext_default',
#         '-d', '"I will not ever vaccinate my children"'
#     ])
#     assert exit_status == 0
