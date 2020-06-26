import argparse
import sys, os
import logging

sys.path.append('..')

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

    def build(self):
        from text_classification.utils.deploy_helpers import build
        parser = ArgParseDefault(description='Build Docker image from trained model')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        parser.add_argument('-p', '--project', type=str, required=True, dest='project', help='Name of project')
        parser.add_argument('-m', '--model_type', choices=['fasttext'], type=str, default='fasttext', dest='model_type', help='Model type')
        args = parser.parse_args(sys.argv[2:])
        build(args.run, args.project, args.model_type)

    def push(self):
        from text_classification.utils.deploy_helpers import push
        parser = ArgParseDefault(description='Push Docker image to AWS ECR')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        parser.add_argument('-p', '--project', type=str, required=True, dest='project', help='Name of project')
        parser.add_argument('-m', '--model_type', choices=['fasttext'], type=str, default='fasttext', dest='model_type', help='Model type')
        args = parser.parse_args(sys.argv[2:])
        push(args.run, args.project, args.model_type)

    def build_and_push(self):
        from text_classification.utils.deploy_helpers import build_and_push
        parser = ArgParseDefault(description='Build and push Docker image to AWS ECR')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        parser.add_argument('-p', '--project', type=str, required=True, dest='project', help='Name of project')
        parser.add_argument('-m', '--model_type', choices=['fasttext'], type=str, default='fasttext', dest='model_type', help='Model type')
        args = parser.parse_args(sys.argv[2:])
        build_and_push(args.run, args.project, args.model_type)

    def run_local(self):
        from text_classification.utils.deploy_helpers import run_local
        parser = ArgParseDefault(description='Runs build locally')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        parser.add_argument('-p', '--project', type=str, required=True, dest='project', help='Name of project')
        parser.add_argument('-m', '--model_type', choices=['fasttext'], type=str, default='fasttext', dest='model_type', help='Model type')
        args = parser.parse_args(sys.argv[2:])
        run_local(args.run, args.project, args.model_type)

    def create_model(self):
        from text_classification.utils.deploy_helpers import create_model_and_configuration
        parser = ArgParseDefault(description='Creates Sagemaker model and endpoint configuration')
        parser.add_argument('-r', '--run', type=str, required=True, dest='run', help='Name of run')
        parser.add_argument('-p', '--project', type=str, required=True, dest='project', help='Name of project')
        parser.add_argument('-q', '--question_tag', type=str, required=True, dest='question_tag', help='Question tag')
        parser.add_argument('-m', '--model_type', choices=['fasttext'], type=str, default='fasttext', dest='model_type', help='Model type')
        parser.add_argument('-i', '--instance-type', type=str, default='ml.t2.medium', dest='instance_type', help='Instance type, check https://aws.amazon.com/sagemaker/pricing/instance-types/')
        args = parser.parse_args(sys.argv[2:])
        create_model_and_configuration(args.run, args.project, args.question_tag, args.model_type, args.instance_type)

if __name__ == '__main__':
    ArgParse()
