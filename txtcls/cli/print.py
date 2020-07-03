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
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-n', '--num_samples', type=int, default=10,
        help='number of misclassifications printed per sample')
    parser.set_defaults(
        func=lambda args: helpers.print_misclassifications(**vars(args)))
