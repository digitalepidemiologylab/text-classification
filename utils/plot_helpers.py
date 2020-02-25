import os
import logging
import sklearn.metrics
import pandas as pd
from utils.helpers import find_project_root
import seaborn as sns
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def plot_confusion_matrix(run):
    f_path = os.path.join(find_project_root(), 'output', run)
    if not os.path.isdir(f_path):
        raise FileNotFoundError(f'Could not find run directory {f_path}')
    test_output_file = os.path.join(find_project_root(), 'output', run, 'test_output.csv')
    if not os.path.isfile(test_output_file):
        raise FileNotFoundError(f'No file {test_output_file} found for run {run}. Pass the option `write_test_output: true` when training the model.')
    df = pd.read_csv(test_output_file)
    cnf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    sns.heatmap(cnf_matrix, ax=ax, annot=True)
    save_fig(fig, f_path, 'confusion_matrix')

def save_fig(fig, folder_path, name, plot_formats=['png'], dpi=300):
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.join(folder_path, f_name)
    for fmt in plot_formats:
        if not fmt == 'tiff':
            f_path = f_name(fmt)
            logger.info('Writing figure file {}'.format(os.path.abspath(f_path)))
            fig.savefig(f_name(fmt), bbox_inches='tight', dpi=dpi)
