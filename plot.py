import argparse
import sys, os
import logging

USAGE_DESC = """
python plot.py <command> [<args>]

Available commands:
  confusion_matrix             Plot confusion matrix for a specific run
  label_distribution           Plot label distributions (full and unambiguous)
"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

class ArgParseDefault(argparse.ArgumentParser):
    """Simple wrapper which shows defaults in help"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class ArgParse(object):
    def __init__(self):
        parser = ArgParseDefault(
                description='',
                usage=USAGE_DESC)
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
        args = parser.parse_args(sys.argv[2:3])
        plot_confusion_matrix(args.run)

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
