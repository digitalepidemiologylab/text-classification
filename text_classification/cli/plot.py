"""CLI plotting module."""

import sys
import logging

from ..utils.misc import ArgParseDefault, add_bool_arg
from ..utils import plot_helpers as helpers

USAGE_DESC = """
python plot.py <command> [<args>]

Available commands:
  confusion_matrix             Plot confusion matrix for a specific run
  compare_runs                 Compare performan between runs (horizontal bar plot)
  label_distribution           Plot label distributions (full and unambiguous)
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def confusion_matrix():
    """Plots confusion matrix."""
    parser = ArgParseDefault(description=confusion_matrix.__doc__)
    parser.add_argument(
        '-r', '--run',
        required=True, dest='run', type=str,
        help='Name of run')
    add_bool_arg(
        parser, 'log_scale', default=False, help='Show values in log scale')
    add_bool_arg(
        parser, 'normalize', default=False, help='Normalize counts')
    args = parser.parse_args(sys.argv[2:])
    helpers.plot_confusion_matrix(args.run, args.log_scale, args.normalize)


def compare_runs():
    """Compares performance between runs (horizontal bar plot)."""
    parser = ArgParseDefault(description=compare_runs.__doc__)
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
    args = parser.parse_args(sys.argv[2:])
    helpers.plot_compare_runs(args.runs, args.performance_scores)


def label_distribution():
    """Plots label distribution."""
    parser = ArgParseDefault(description=label_distribution.__doc__)
    parser.add_argument(
        '-d', '--data-path',
        required=True, type=str,
        help='Data path')
    args = parser.parse_args(sys.argv[2:])
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
