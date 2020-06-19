import argparse
import sys, os
import logging
from utils.misc import ArgParseDefault

USAGE_DESC = """
python plot.py <command> [<args>]

Available commands:
  confusion_matrix             Plot confusion matrix for a specific run
  compare_runs                 Compare performan between runs (horizontal bar plot)
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
        args = parser.parse_args(sys.argv[2:])
        plot_confusion_matrix(args.run)

    def compare_runs(self):
        from utils.plot_helpers import plot_compare_runs
        parser = ArgParseDefault(description='Compare performan between runs (horizontal bar plot)')
        parser.add_argument('-r', '--runs', type=str, required=True, nargs='+', help='Name of runs to compare. Optional: Specify as run_name:figure_name to show different name in figure')
        parser.add_argument('-s', '--performance_scores', type=list, default=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], nargs='+', help='Scores to plot')
        args = parser.parse_args(sys.argv[2:])
        plot_compare_runs(args.runs, args.performance_scores)

if __name__ == '__main__':
    ArgParse()
