"""CLI main module."""

import os
import logging

import joblib

from ..utils import helpers
from ..utils import deploy_helpers
from ..utils.list_runs import ListRuns


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def preprocess(parser):
    """Preprocesses data."""
    parser.add_argument(
        '-c', '--config', default='config.json', metavar='C',
        help='name/path of configuration file')

    def _preprocess(args):
        config_reader = helpers.ConfigReader()
        config = config_reader.parse_config(args.config, mode='preprocess')
        for run_config in config.runs:
            helpers.preprocess(run_config)

    parser.set_defaults(func=_preprocess)


def split(parser):
    """Splits annotated data into training and test data sets."""
    parser.add_argument(
        '-n', '--name', type=str, required=True,
        help='name of dataset or file path')
    parser.add_argument(
        '-s', '--test-size', type=float, default=0.2,
        help='fraction of test size')
    parser.add_argument(
        '--balanced-labels', default=False, action='store_true',
        help='ensure equal label balance')
    parser.add_argument(
        '--label-tags', default=[], nargs='+',
        help='only select examples with certain label tags')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random state split')
    parser.set_defaults(
        func=lambda args: helpers.train_test_split(**vars(args)))


def train(parser):
    """Trains model based on config.

    The following config keys can/should be present in the config file
    (in runs or params).

    Args:
        name (required): Unique name of the run
        model (required): One of the available models
            (e.g. fasttext, bert, etc.)
        train_data (required): Path to training data
            (if only filename is provided it should be located under `data/`)
        test_data (required): Path to test data
            (if only filename is provided it should be located under `data/`)
        augment_data: Path to augment data
            (if only filename is provided it should be located under `data/`)
        overwrite: If run output folder is already present, wipe it and
            create new folder
        write_test_output: Write output csv of test evaluation.
            Default: ``False``.
        test_only: Runs test file only and skips training. Default: ``False``.
    """
    parser.add_argument(
        '-c', '--config', default='config.json', metavar='C',
        help='name/path of configuration file')
    parser.add_argument(
        '--parallel', action='store_true',
        help='run in parallel (only recommended for CPU-training)')

    def _train(args):
        config_reader = helpers.ConfigReader()
        config = config_reader.parse_config(args.config, mode='train')
        if len(config.runs) > 1 and args.parallel:
            num_cpus = max(os.cpu_count() - 1, 1)
            parallel = joblib.Parallel(n_jobs=num_cpus)
            train_delayed = joblib.delayed(helpers.train)
            parallel((train_delayed(run_config) for run_config in config.runs))
        else:
            for run_config in config.runs:
                helpers.train(run_config)

    parser.set_defaults(func=_train)


def predict(parser):
    """Predicts classes based on a config file and input data and
    output predictions.
    """
    parser.add_argument(
        '-r', '--run', type=str, required=True, dest='run_name',
        help='name of run')
    parser.add_argument(
        '-p', '--path', type=str, default=None,
        help='input path of data file for predictions')
    parser.add_argument(
        '-d', '--data', type=str, default=None,
        help='input text as argument (ignored if path is given)')
    parser.add_argument(
        '-o', '--output-folder', type=str, default='predictions',
        help='output folder')
    parser.add_argument(
        '-f', '--output-formats', default=['csv'],
        nargs='+', choices=['csv', 'json'],
        help='output folder')
    parser.add_argument(
        '--col', type=str, default='text',
        help="in case input is a CSV, use this as the column name; "
             "if the input is a .txt file this option won't have any effect")
    parser.add_argument(
        '--output-cols',
        type=str, default='labels,probabilities,label,probability',
        help="output columns, provide as comma-separted string; "
             "only applies to CSV format")
    parser.add_argument(
        '--no-file-output', default=False, action='store_true',
        help='do not write output file '
             '(default: write output file to `./predictions/` folder)')
    parser.add_argument(
        '--in-parallel', default=False, action='store_true',
        help='run predictions in parallel '
             '(only recommmended for CPU-based models)')
    parser.add_argument(
        '--verbose', default=False, action='store_true',
        help='print predictions')
    parser.set_defaults(func=lambda args: helpers.predict(**vars(args)))


def generate_config(parser):
    """Generates config for grid search hyperparameter search."""
    parser.add_argument(
        '--name', type=str, required=True,
        help='global name prefix and name of output file')
    parser.add_argument(
        '--train-data', type=str,
        help='train data path')
    parser.add_argument(
        '--test-data', type=str,
        help='test data path')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='model')
    parser.add_argument(
        '-p', '--params', nargs='*', default=[],
        help="""
        arbitrary list of model params for grid search.
        Format: 'key:modifier:values', where
        - 'key'       hyperparameter name,
        - 'modifier'  can be either
            - 'val' (individual values),
            - 'lin' (linspace), or
            - 'log' (logspace)
        - 'values'    params for the lin/log space.
        Examples:
        - 'num_epochs:val:2,3'
        - 'learning_rate:log:-6,-2,4'
        """)
    parser.add_argument(
        '-g', '--globals', dest='global_params', nargs='*', default=[],
        help="list of global params which will be passed "
             "to all runs.\nFormat: 'key:value'")
    parser.set_defaults(
        func=lambda args: helpers.generate_config(**vars(args)))


def generate_text(parser):
    """Generates text."""
    parser.add_argument(
        '--seed', type=str, required=True,
        help='text generator seed')

    def raise_not_implemented(args):
        raise NotImplementedError

    parser.set_defaults(
        func=raise_not_implemented)

    # args, other_args = parser.parse_known_args(sys.argv[2:])
    # for arg in other_args:
    #     if arg.startswith(("-", "--")):
    #         parser.add_argument(arg)
    # args = parser.parse_args(sys.argv[2:])
    # text = helpers.generate_text(**vars(args))
    # print(text)


def augment(parser):
    """Augments training data."""
    parser.add_argument(
        '-n', '--num', type=int, default=10, dest='n',
        help='number of tweets to generate text from')
    parser.add_argument(
        '--min-tokens', type=int, default=8)
    parser.add_argument(
        '-s', '--source', type=str, default='training_data',
        help='source for seed data (training_data or seed_data)')
    parser.add_argument(
        '-r', '--repeats', type=int, default=1,
        help='number of repeated times a single seed should be used')
    parser.add_argument(
        '-c', '--contains',
        type=str, default='', dest='should_contain_keyword',
        help='only collect sentences which contain keyword')
    parser.add_argument(
        '--n-sentences-after-seed', type=int, default=8,
        help='consider only first n predicted sentences after seed')
    parser.add_argument(
        '--verbose', action='store_true',
        help='verbose output')
    parser.set_defaults(
        func=lambda args: helpers.augment_training_data(**vars(args)))


def pretrain(parser):
    """Pretrains model based on config.

    The following config keys can/should be present in the config file
    (in runs or params).

    Args:
        name (required): Unique name of the run
        model (required): One of the models which can be pretrained
            (``'fasttext'`` or ``'bert'``)
        pretrain_data (required): Path to unannotated data.
            A csv with a text column
            (if only filename is provided it should be located under `data/`)
        pretrain_test_data: If provided will calculate perplexity
        overwrite: Wipe existing pretrained model with same name
        etc: all additional model-specific parameters
    """
    parser.add_argument(
        '-c', '--config', metavar='C', default='config.json',
        help='name/path of configuration file')

    def _pretrain(args):
        config_reader = helpers.ConfigReader()
        config = config_reader.parse_pretrain_config(args.config)
        for run_config in config.runs:
            helpers.pretrain(run_config)

    parser.set_defaults(func=_pretrain)


def learning_curve(parser):
    """Computes learning curve for model.

    The following keys can/should be present in the config file.

    Args:
        name (required): Unique prefix of all learning curve runs
            (each name will be extended with a run index ``_run_i``)
        model (required): One of the available models
        train_data (required)
        test_data (required)
        learning_curve_fractions_linspace: Argument to ``np.linspace``
            which sets the fraction of training data which should be used.
            Default: [0, 1, 20].
        learning_curve_repetitions: Number of times each fraction will be
            trained and evaluated (each fraction will be randomly sampled).
            Default: 1
        etc: all parameters which are also available for ``train``

    Note that all fractions will contain at least one example of each
    available class (so that fraction 0 there will be at least num_classes
    training samples present).
    Output configs will contain keys `learning_curve_fraction` and
    `learning_curve_num_samples` indicating the portion of training data
    which was used, as well as a unique identifier `learning_curve_id` and
    a unique run index `learning_curve_index` for each run and
    a `learning_curve_repetition_index`, being unique for each repetition group.
    """
    parser.add_argument(
        '-c', '--config', metavar='C',
        dest='config_path', default='config.json',
        help='name/path of configuration file')
    parser.set_defaults(func=lambda args: helpers.learning_curve(**vars(args)))


def optimize(parser):
    """Performs hyperparameter optimization (requires hypteropt package).

    The following keys can/should be present in the config file.

    Args:
        name (required): Unique prefix of all optimization output
            will be stored under
        model (required): One of the available models
        train_data (required)
        test_data (required)
        optimize_space (required): List of hyperparameter-objects to optimize.
            A hyperparameter-object is a dictionary with keys ``param``
            (name of the hyperparameter), ``type`` (choice|uniform|normal),
            and ``values``.
            (For choice list of choices, otherwise a list of arguments
            to be passed to function, can be stringified python code.)
        optimize_max_eval: Maximum number of iterations (default 10)
        optimize_keep_models: Whether to keep model for each iteration.
            Default: ``False``.
    """
    parser.add_argument(
        '-c', '--config', metavar='C',
        dest='config_path', default='config.json',
        help='name/path of configuration file')
    parser.set_defaults(
        func=lambda args: helpers.optimize(**vars(args)))


def ls(parser):
    """List trained models."""
    parser.add_argument(
        '-m', '--model', type=str, default=None,
        help='only show certain models')
    parser.add_argument(
        '-r', '--run-patterns', type=str, default=('*', ), nargs='+',
        help='filter by run name patterns')
    parser.add_argument(
        '-p', '--params', type=str, default=None, nargs='+',
        help='display certain hyperparameters instead of default ones')
    parser.add_argument(
        '-a', '--averaging', type=str, default='macro',
        choices=['micro', 'macro', 'weighted'],
        help='precision/recall/f1 averaging mode')
    parser.add_argument(
        '-t', '--top', type=int, default=40,
        help='maximum number of models to show, use -1 to show all')
    parser.add_argument(
        '--metrics', type=str, default=['f1', 'precision', 'recall', 'accuracy'],
        nargs='+', choices=['accuracy', 'f1', 'precision', 'recall'],
        help='metrics to display (also defines sorting order)')
    parser.add_argument(
        '--names-only', action='store_true',
        help='only list names')
    parser.add_argument(
        '--all-params', action='store_true',
        help='show all params')
    parser.add_argument(
        '--sort-list', type=str, default=None, nargs='+',
        help='list of parameters for multi-column sorting')
    parser.set_defaults(
        func=lambda args: ListRuns().list_runs(**vars(args)))


def deploy(parser):
    """Makes trained models available to AWS Sagemaker.

    Performs the following steps:
    1) Create local Docker image (requires Docker)
    2) Push image to AWS ECR (requires AWS user with access to ECR)
    3) Push model artefacts to a S3 bucket "crowdbreaks-sagemaker"
    (requires AWS user with access to this bucket)
    4) Create AWS Sagemaker model (requires an AWS role "crowdbreaks-sagemaker"
    with Sagemaker access)
    5) Create AWS Sagemaker endpoint configuration, specifying
    the instance type to use for this model
    After this the endpoint still needs to be created manually.
    """
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
        '-m', '--model_type', type=str, default='fasttext',
        choices=['fasttext'],
        help='model type')
    parser.add_argument(
        '-i', '--instance-type', type=str, default='ml.t2.medium',
        help='instance type, check https://aws.amazon.com/sagemaker/pricing/instance-types/')
    parser.set_defaults(
        func=lambda args: deploy_helpers.deploy(**vars(args)))
