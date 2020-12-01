"""CLI deployment moddule."""

import logging

from ..utils import deploy_helpers as helpers

logger = logging.getLogger(__name__)


def build(parser):
    """Builds Docker image from trained model."""
    parser.add_argument(
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-p', '--project', type=str, required=True,
        help='name of project')
    parser.add_argument(
        '-m', '--model-type',
        type=str, default='fasttext', choices=['fasttext'],
        help='model type')
    parser.set_defaults(
        func=lambda args: helpers.build(**vars(args)))


def push(parser):
    """Pushes Docker image to AWS ECR."""
    parser.add_argument(
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-p', '--project', type=str, required=True,
        help='name of project')
    parser.add_argument(
        '-m', '--model-type',
        type=str, default='fasttext', choices=['fasttext'],
        help='model type')
    parser.set_defaults(
        func=lambda args: helpers.push(**vars(args)))


def build_and_push(parser):
    """Builds and pushes Docker image to AWS ECR."""
    parser.add_argument(
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-p', '--project', type=str, required=True,
        help='name of project')
    parser.add_argument(
        '-m', '--model-type',
        type=str, default='fasttext', choices=['fasttext'],
        help='model type')
    parser.set_defaults(
        func=lambda args: helpers.build_and_push(**vars(args)))


def run_local(parser):
    """Runs build locally."""
    parser.add_argument(
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-p', '--project', type=str, required=True,
        help='name of project')
    parser.add_argument(
        '-m', '--model-type',
        type=str, default='fasttext', choices=['fasttext'],
        help='model type')
    parser.set_defaults(
        func=lambda args: helpers.run_local(**vars(args)))


def create_model(parser):
    """Creates Sagemaker model and endpoint configuration."""
    parser.add_argument(
        '-r', '--run', type=str, required=True,
        help='name of run')
    parser.add_argument(
        '-p', '--project', type=str, required=True,
        help='name of project')
    parser.add_argument(
        '-q', '--question-tag', type=str, required=True,
        help='question tag')
    parser.add_argument(
        '-m', '--model-type',
        type=str, default='fasttext', choices=['fasttext'],
        help='model type')
    parser.add_argument(
        '-i', '--instance-type', type=str, default='ml.t2.medium',
        help='instance type, check https://aws.amazon.com/sagemaker/pricing/instance-types/')
    parser.set_defaults(
        func=lambda args: helpers.create_model_and_configuration(**vars(args)))
