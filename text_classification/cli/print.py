"""CLI printing module."""

import sys
import logging

from ..utils.misc import ArgParseDefault
from ..utils import print_helpers as helpers

USAGE_DESC = """
python print.py <command> [<args>]

Available commands:
  misclassifications             Print sample of misclassifications for given run
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def misclassifications():
    """Prints misclassifications."""
    parser = ArgParseDefault(description=misclassifications.__doc__)
    parser.add_argument(
        '-r', '--run',
        required=True, dest='run', type=str,
        help='Name of run')
    parser.add_argument(
        '-n', '--num_samples',
        required=False, type=int, default=10,
        help='Number of misclassifications printed per sample')
    args = parser.parse_args(sys.argv[2:])
    helpers.print_misclassifications(args.run, args.num_samples)
