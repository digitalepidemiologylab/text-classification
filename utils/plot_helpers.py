import os
import joblib
import logging
import sklearn.metrics
import pandas as pd
from utils.helpers import find_project_root, get_label_mapping
from utils.list_runs import ListRuns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

def plot_confusion_matrix(run, log_scale, normalize):
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
    fmt = 'd'
    f_name = run
    if log_scale:
        df = np.log(df + 1)
        fmt = '1.1f'
        f_name += '_log_scale'
    if normalize:
        df = df.divide(df.sum(axis=1), axis=0)
        fmt = '1.1f'
        f_name += '_normalized'
    sns.heatmap(df, ax=ax, annot=True, fmt=fmt, annot_kws={"fontsize": 8})
    ax.set(xlabel='predicted label', ylabel='true label')
    save_fig(fig, 'confusion_matrix', f_name)

def plot_compare_runs(runs, performance_scores, order_by):
    df = []
    run_dict = {}
    for run in runs:
        if ':' in run:
            run_name, alt_name = run.split(':')
            run_dict[run_name] = alt_name
        else:
            run_dict[run] = run
    for run, alt_name in run_dict.items():
        _df = ListRuns.collect_results(run=run)
        _df['name'] = alt_name
        if len(_df) == 0:
            raise FileNotFoundError(f'Could not find the run "{run}" in ./output/')
        elif len(_df) > 1:
            raise ValueError(f'Run name "{run}" is not unique. Found {len(_df):,} matching runs for this pattern.')
        df.append(_df)
    df = pd.concat(df)
    # collect scores and vlines
    scores = []
    vlines = []
    for score in performance_scores:
        if ':' in score:
            score, _ = score.split(':')
            vlines.append(score)
        scores.append(score)
    scores = list(set(scores))
    vlines = list(set(vlines))
    # melt
    df = df[['name', *scores]].melt(id_vars=['name'], var_name='performance', value_name='score')
    # order
    hue_order = None
    if order_by is not None:
        order = df[df.performance == order_by].sort_values('score').name.tolist()
        hue_order = df[df.name == order[-1]].sort_values('score').performance.tolist()
        hue_order.remove(order_by)
        hue_order.insert(0, order_by)
    # plot
    g = sns.catplot(x='score', y='name', hue='performance', kind='bar', orient='h', ci=None, aspect=2, palette='colorblind', data=df, order=order, hue_order=hue_order)
    for vline_score in vlines:
        vline_values  = df[df.performance == vline_score]['score'].values
        for v in vline_values:
            g.ax.axvline(v, ls='--', c='.1', lw=.5)
    fig = plt.gcf()
    save_fig(fig, 'compare_runs', '-'.join(run_dict.values()))

def plot_label_distribution(data_path, mode='test', label='category', merged=True):
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
    save_fig(fig, 'label_distribution', data_dir)
    file_name = '_'.join(config_dir + [mode, 'label-distribution'])
    pics_dir = os.path.join(data_path, 'pics')
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)
    save_fig(fig, pics_dir, file_name)

def save_fig(fig, fig_type, name, plot_formats=['png'], dpi=300):
    folder = os.path.join(find_project_root(), 'plots', fig_type)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.abspath(os.path.join(folder, f_name))
    for fmt in plot_formats:
        f_path = f_name(fmt)
        logger.info(f'Writing figure file {f_path}')
        fig.savefig(f_path, bbox_inches='tight', dpi=dpi)
