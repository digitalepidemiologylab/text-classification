"""CLI entry-point functions."""

import sys

from ..utils.misc import ArgParseDefault
from . import main as main_
from . import deploy as deploy_
from . import plot as plot_
from . import print as print_


func_to_mod = {
    'main_cli': 'main_',
    'deploy_cli': 'deploy_',
    'plot_cli': 'plot_',
    'print_cli': 'print_'
}


def entry_point(func):
    def wrapper():
        parser = ArgParseDefault(
            usage=globals()[func_to_mod[func.__name__]].USAGE_DESC)
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        try:
            getattr(globals()[func_to_mod[func.__name__]], args.command)()
        except AttributeError:
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
    return wrapper


@entry_point
def main_cli():
    pass


@entry_point
def deploy_cli():
    pass


@entry_point
def plot_cli():
    pass


@entry_point
def print_cli():
    pass
