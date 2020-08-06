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
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '--log-scale', default=False, action='store_true',
        help='show values in log scale')
    parser.add_argument(
        '--normalize-sum', default=False, action='store_true',
        help='normalize counts')
    parser.add_argument(
        '--normalize-test', default=False, action='store_true',
        help='normalize counts')
    parser.add_argument(
        '--figsize_y', type=float, default=6,
        help='figsize y')
    parser.add_argument(
        '--figsize_x', type=int, default=9,
        help='figsize x')
    parser.add_argument(
        '--stylesheet', type=str, default=None,
        help='stylesheet')
    parser.add_argument(
        '--vmin', type=int, default=None,
        help='colorbar vmin')
    parser.add_argument(
        '--vmax', type=int, default=None,
        help='colorbar vmax')
    parser.add_argument(
        '--vmin-norm', type=int, default=None,
        help='colorbar vmin')
    parser.add_argument(
        '--vmax-norm', type=int, default=None,
        help='colorbar vmax')
    parser.add_argument(
        '--plot-formats', type=str, default=['png'], nargs='+',
        help='plot formats')
    parser.set_defaults(
        func=lambda args: helpers.plot_confusion_matrix(**vars(args)))


def compare_runs(parser):
    """Compares performance between runs (horizontal bar plot)."""
    parser.add_argument(
        '-r', '--runs', type=str, required=True, nargs='+',
        help="""
        name of runs to compare.
        Optional: Specify as 'run_name:figure_name'
        to show different name in figure
        """)
    parser.add_argument(
        '-s', '--performance_scores', type=list, nargs='+',
        default=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        help='scores to plot')
    parser.set_defaults(
        func=lambda args: helpers.plot_compare_runs(**vars(args)))


def label_distribution(parser):
    """Plots label distribution."""
    parser.add_argument(
        '-d', '--data-path', type=str, required=True,
        help='data path')

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
