"""CLI plotting module."""

import logging

from ..utils import plot_helpers as helpers


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def confusion_matrix(parser):
    """Plots confusion matrix."""
    parser.add_argument(
        '-r', '--run',
        required=True, dest='run', type=str,
        help='Name of run')
    parser.add_argument(
        '--log-scale',
        dest='log_scale', action='store_true', default=False,
        help='Show values in log scale')
    parser.add_argument(
        '--normalize',
        dest='normalize', action='store_true', default=False,
        help='Normalize counts')
    parser.set_defaults(
        func=lambda args: helpers.plot_confusion_matrix(**vars(args)))


def compare_runs(parser):
    """Compares performance between runs (horizontal bar plot)."""
    parser.add_argument(
        '-r', '--runs',
        required=True, type=str, nargs='+',
        help='Name of runs to compare. '
             'Optional: Specify as run_name:figure_name to show '
             'different name in figure')
    parser.add_argument(
        '-s', '--performance_scores',
        type=list, nargs='+',
        default=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        help='Scores to plot')
    parser.set_defaults(
        func=lambda args: helpers.plot_compare_runs(**vars(args)))


def label_distribution(parser):
    """Plots label distribution."""
    parser.add_argument(
        '-d', '--data-path',
        required=True, type=str,
        help='Data path')

    def _label_distribution(args):
        config_dict = {}
        config_dict['mode'] = ['train', 'test']
        config_dict['label'] = ['category', 'type']
        config_dict['merged'] = [True, False]
        for mode in config_dict['mode']:
            for label in config_dict['label']:
                for merged in config_dict['merged']:
                    helpers.plot_label_distribution(
                        args.data_path,
                        mode=mode, label=label, merged=merged)

    parser.set_defaults(func=_label_distribution)
