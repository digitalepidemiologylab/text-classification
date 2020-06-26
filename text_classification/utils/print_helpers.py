import os
import pandas as pd
from text_classification.utils.helpers import find_project_root
import logging


logger = logging.getLogger(__name__)

def print_misclassifications(run, num_samples):
    f_path = os.path.join(find_project_root(), 'output', run)
    if not os.path.isdir(f_path):
        raise FileNotFoundError(f'Could not find run directory {f_path}')
    test_output_file = os.path.join(find_project_root(), 'output', run, 'test_output.csv')
    if not os.path.isfile(test_output_file):
        raise FileNotFoundError(f'No file {test_output_file} found for run {run}. Pass the option `write_test_output: true` when training the model.')
    df = pd.read_csv(test_output_file)
    for label, grp in df.groupby('label'):
        misclassifications = grp[grp.prediction != label]
        num_misclassifications = len(misclassifications)
        print(f'True label: {label.ljust(10)} (num misclassifications: {num_misclassifications:,})')
        if num_misclassifications == 0:
            print('<No misclassifications to show>')
            continue
        show_samples = min(num_misclassifications, num_samples)
        for i, row in misclassifications.sample(show_samples).iterrows():
            print(f'Predicted: {row.prediction.ljust(20)} | {row.text}')
        print('-'*40)
