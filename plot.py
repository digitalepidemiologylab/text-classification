import argparse
import sys, os
import logging
from utils.misc import ArgParseDefault, add_bool_arg

USAGE_DESC = """
python plot.py <command> [<args>]

Available commands:
  confusion_matrix             Plot confusion matrix for a specific run
  compare_runs                 Compare performan between runs (horizontal bar plot)
  label_distribution           Plot label distributions (full and unambiguous)
"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

class ArgParse():
    def __init__(self):
        parser = ArgParseDefault(usage=USAGE_DESC)
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        getattr(self, args.command)()

    def confusion_matrix(self):
        from utils.plot_helpers import plot_confusion_matrix
        parser = ArgParseDefault(description='Plot confusion matrix')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        add_bool_arg(parser, 'log_scale', default=False, help='Show values in log scale')
        add_bool_arg(parser, 'normalize', default=False, help='Normalize counts')
        args = parser.parse_args(sys.argv[2:])
        plot_confusion_matrix(args.run, args.log_scale, args.normalize)

    def compare_runs(self):
        from utils.plot_helpers import plot_compare_runs
        parser = ArgParseDefault(description='Compare performan between runs (horizontal bar plot)')
        parser.add_argument('-r', '--runs', type=str, required=True, nargs='+', help='Name of runs to compare. Optional: Specify as run_name:figure_name to show different name in figure')
        parser.add_argument('-s', '--performance_scores', type=list, default=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], nargs='+', help='Scores to plot')
        args = parser.parse_args(sys.argv[2:])
        plot_compare_runs(args.runs, args.performance_scores)

    def label_distribution(self):
        from utils.plot_helpers import plot_label_distribution
        parser = ArgParseDefault(description='Plot label distribution')
        parser.add_argument('-d', '--data-path', type=str, required=True, help='Data path')
        args = parser.parse_args(sys.argv[2:4])
        config_dict = {}
        config_dict['mode'] = ['train', 'test']
        config_dict['label'] = ['category', 'type']
        config_dict['merged'] = [True, False]
        for mode in config_dict['mode']:
            for label in config_dict['label']:
                for merged in config_dict['merged']:
                    plot_label_distribution(
                        args.data_path,
                        mode=mode, label=label, merged=merged)

if __name__ == '__main__':
    ArgParse()
