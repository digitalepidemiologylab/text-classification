import subprocess

import pytest


def test_preprocess():
    exit_status = subprocess.call([
        'txtcls', 'main', 'preprocess', '-c',
        'configs/config.preprocess.twitter-hate-speech.json'
    ])
    assert exit_status == 0


def test_train():
    exit_status = subprocess.call([
        'txtcls', 'main', 'train',
        '-c', 'configs/config.train.twitter-hate-speech.json'
    ])
    assert exit_status == 0
