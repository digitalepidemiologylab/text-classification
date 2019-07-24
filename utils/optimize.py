import os
import json
from utils.config_reader import ConfigReader
import pandas as pd
import numpy as np

class Optimize():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_reader = ConfigReader()

    def init(self):
        pass
