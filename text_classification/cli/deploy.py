"""CLI deployment moddule."""

import sys
import logging

from ..utils.misc import ArgParseDefault
from ..utils import deploy_helpers as helpers

USAGE_DESC = """
python deploy.py <command> [<args>]

These are helpers to run the individual steps of model deployment. You can run all these steps at once using `python main.py deploy`.

Available commands:
  build            Dockerize trained model
  push             Push dockerized model to AWS ECR
  build_and_push   Runs both build and push
  run_local        Builds image and runs it locally
  create_model     Creates Sagemaker model and endpoint configuration
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def build():
    """Builds Docker image from trained model."""
    parser = ArgParseDefault(description=build.__doc__)
    parser.add_argument(
        '-r', '--run',
        type=str, required=True, dest='run',
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        type=str, required=True, dest='project',
        help='Name of project')
    parser.add_argument(
        '-m', '--model_type',
        choices=['fasttext'], type=str, default='fasttext', dest='model_type',
        help='Model type')
    args = parser.parse_args(sys.argv[2:])
    helpers.build(args.run, args.project, args.model_type)


def push():
    """Pushes Docker image to AWS ECR."""
    parser = ArgParseDefault(description=push.__doc__)
    parser.add_argument(
        '-r', '--run',
        type=str, required=True, dest='run',
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        type=str, required=True, dest='project',
        help='Name of project')
    parser.add_argument(
        '-m', '--model_type',
        choices=['fasttext'], type=str, default='fasttext', dest='model_type',
        help='Model type')
    args = parser.parse_args(sys.argv[2:])
    helpers.push(args.run, args.project, args.model_type)


def build_and_push():
    """Builds and pushes Docker image to AWS ECR."""
    parser = ArgParseDefault(description=build_and_push.__doc__)
    parser.add_argument(
        '-r', '--run',
        type=str, required=True, dest='run',
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        type=str, required=True, dest='project',
        help='Name of project')
    parser.add_argument(
        '-m', '--model_type',
        choices=['fasttext'], type=str, default='fasttext', dest='model_type',
        help='Model type')
    args = parser.parse_args(sys.argv[2:])
    helpers.build_and_push(args.run, args.project, args.model_type)


def run_local():
    """Runs build locally."""
    parser = ArgParseDefault(description=run_local.__doc__)
    parser.add_argument(
        '-r', '--run',
        type=str, required=True, dest='run',
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        type=str, required=True, dest='project',
        help='Name of project')
    parser.add_argument(
        '-m', '--model_type',
        choices=['fasttext'], type=str, default='fasttext', dest='model_type',
        help='Model type')
    args = parser.parse_args(sys.argv[2:])
    helpers.run_local(args.run, args.project, args.model_type)


def create_model():
    """Creates Sagemaker model and endpoint configuration."""
    parser = ArgParseDefault(description=create_model.__doc__)
    parser.add_argument(
        '-r', '--run',
        type=str, required=True, dest='run',
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        type=str, required=True, dest='project',
        help='Name of project')
    parser.add_argument(
        '-q', '--question_tag',
        type=str, required=True, dest='question_tag',
        help='Question tag')
    parser.add_argument(
        '-m', '--model_type',
        choices=['fasttext'], type=str, default='fasttext', dest='model_type',
        help='Model type')
    parser.add_argument(
        '-i', '--instance-type',
        type=str, default='ml.t2.medium', dest='instance_type',
        help='Instance type, check https://aws.amazon.com/sagemaker/pricing/instance-types/')
    args = parser.parse_args(sys.argv[2:])
    helpers.create_model_and_configuration(
        args.run, args.project, args.question_tag,
        args.model_type, args.instance_type)
