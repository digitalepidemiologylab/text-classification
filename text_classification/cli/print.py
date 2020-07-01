"""CLI printing module."""

import logging

from ..utils import print_helpers as helpers


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def misclassifications(parser):
    """Prints misclassifications."""
    parser.add_argument(
        '-r', '--run',
        required=True, dest='run', type=str,
        help='Name of run')
    parser.add_argument(
        '-n', '--num_samples',
        required=False, type=int, default=10,
        help='Number of misclassifications printed per sample')
    parser.set_defaults(
        func=lambda args: helpers.print_misclassifications(**vars(args)))
