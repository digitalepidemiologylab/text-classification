import os
import joblib
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
    label_mapping = get_label_mapping(f_path)
    labels = list(label_mapping.keys())
    cnf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction)
    df = pd.DataFrame(cnf_matrix, columns=labels, index=labels)
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    sns.heatmap(df, ax=ax, annot=True, fmt='d', annot_kws={"fontsize":8})
    ax.set(xlabel='predicted label', ylabel='true label')
    save_fig(fig, f_path, 'confusion_matrix')


def plot_label_distribution(
    data_path, mode='test', label='category', merged=True
):
    assert mode in ['train', 'test']
    assert label in ['category', 'type']
    assert type(merged) == bool
    config_dir = [label]
    if merged:
        config_dir.append('merged')

    data_dir = os.path.join(
        data_path, mode, '_'.join(config_dir))
    data_dir_unambiguous = os.path.join(
        data_path, mode, '_'.join(config_dir + ['unambiguous']))
    title = f"{label.capitalize()} {mode.capitalize()} " \
            f"{'Merged' if merged else ''}"

    df = pd.read_csv(os.path.join(data_dir, 'all.csv'))
    df_unambiguous = pd.read_csv(os.path.join(data_dir_unambiguous, 'all.csv'))
    labels = dict(df.label.value_counts())
    labels_unambiguous = dict(df_unambiguous.label.value_counts())
    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    g = sns.barplot(
        x=list(labels.values()), y=list(labels.keys()),
        ax=ax, orient='h', label='Full',
        color=sns.color_palette('muted')[0], edgecolor='w')
    g.set_xscale('log')
    g_unambiguous = sns.barplot(
        x=list(labels_unambiguous.values()),
        y=list(labels_unambiguous.keys()),
        ax=ax, orient='h', label='Unambiguous',
        color=sns.color_palette('bright')[0], edgecolor='w')
    g_unambiguous.set_xscale('log')
    ax.legend(loc='lower right')
    ax.set(title=title, xlabel='Number of samples', ylabel='Label')
    save_fig(fig, data_dir, 'label_distribution')
    file_name = '_'.join(config_dir + [mode, 'label-distribution'])
    pics_dir = os.path.join(data_path, 'pics')
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)
    save_fig(fig, pics_dir, file_name)


def get_label_mapping(run_path):
    label_mapping_path = os.path.join(run_path, 'label_mapping.pkl')
    if not os.path.isfile(label_mapping_path):
        raise FileNotFoundError(f'Could not find label mapping file {label_mapping_path}')
    with open(label_mapping_path, 'rb') as f:
        label_mapping = joblib.load(f)
    return label_mapping

def save_fig(fig, folder_path, name, plot_formats=['png'], dpi=300):
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.join(folder_path, f_name)
    for fmt in plot_formats:
        if not fmt == 'tiff':
            f_path = f_name(fmt)
            logger.info('Writing figure file {}'.format(os.path.abspath(f_path)))
            fig.savefig(f_name(fmt), bbox_inches='tight', dpi=dpi)
