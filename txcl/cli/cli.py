"""CLI entry-point functions."""

from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter,
                      RawTextHelpFormatter)
from inspect import getmembers, isfunction

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


class MixedFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


def entry_point():
    entry_parser = ArgumentParser(prog='txcl', formatter_class=MixedFormatter)
    subparsers = entry_parser.add_subparsers(help='sub-commands')

    main_parser = subparsers.add_parser(
        'main', help='main pipeline')
    deploy_parser = subparsers.add_parser(
        'deploy', help='deployment')
    plot_parser = subparsers.add_parser(
        'plot', help='plotting')
    print_parser = subparsers.add_parser(
        'print', help='printing')

    set_subparsers(main_parser, main_)
    set_subparsers(deploy_parser, deploy_)
    set_subparsers(plot_parser, plot_)
    set_subparsers(print_parser, print_)

    args = entry_parser.parse_args()
    try:
        func = args.func
        del args.func
        func(args)
    except AttributeError:
        print(
            'Please select a function. '
            'To get a list of available functions, use --help flag.')


def set_subparsers(parser, module):
    subparsers = parser.add_subparsers(help='sub-commands')
    funcs = getmembers(module, isfunction)

    def doc_to_help(doc):
        help_str = doc.split('.')[0]
        help_str = help_str[0].lower() + help_str[1:]
        return help_str

    for func_name, func in funcs:
        local_parser = subparsers.add_parser(
            func_name.replace("_", "-"),
            description=func.__doc__,
            formatter_class=MixedFormatter,
            help=doc_to_help(func.__doc__))
        func(local_parser)


# def entry_point_local(func):
#     def wrapper():
#         parser = ArgParseDefault()
#             # usage=globals()[func_to_mod[func.__name__]].USAGE_DESC)
#         parser.add_argument('subcommand', help='Subcommand to run')
#         args = parser.parse_args(sys.argv[1:2])
#         try:
#             getattr(globals()[func_to_mod[func.__name__]], args.command)()
#         except AttributeError:
#             print('Unrecognized command')
#             parser.print_help()
#             sys.exit(1)
#     return wrapper


# @entry_point_local
# def main_cli():
#     pass


# @entry_point_local
# def deploy_cli():
#     pass


# @entry_point_local
# def plot_cli():
#     pass


# @entry_point_local
# def print_cli():
#     pass
