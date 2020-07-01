"""CLI main module."""

import os
import logging

import joblib

from ..utils import helpers
from ..utils import deploy_helpers


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def split(parser):
    """Splits annotated data into training and test data sets."""
    parser.add_argument(
        '-n', '--name',
        type=str, required=True,
        help='Name of dataset or file path')
    parser.add_argument(
        '-s', '--test-size',
        dest='test_size', type=float, required=False, default=0.2,
        help='Fraction of test size')
    parser.add_argument(
        '--balanced-labels',
        dest='balanced_labels', action='store_true', default=False,
        help='Ensure equal label balance')
    parser.add_argument(
        '--label-tags',
        dest='label_tags', required=False, default=[], nargs='+',
        help='Only select examples with certain label tags')
    parser.add_argument(
        '--seed',
        type=int, required=False, default=42,
        help='Random state split')
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
        '-c', '--config',
        metavar='C', required=False, default='config.json',
        help='Name/path of configuration file. Default: config.json')
    parser.add_argument(
        '--parallel',
        required=False, action='store_true',
        help='Run in parallel (only recommended for CPU-training)')

    def _train(args):
        config_reader = helpers.ConfigReader()
        config = config_reader.parse_config(args.config)
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
        '-r', '--run',
        required=True, dest='run_name', type=str, default=None,
        help='Name of run')
    parser.add_argument(
        '-p', '--path',
        required=False, type=str, default=None,
        help='Input path of data file for predictions')
    parser.add_argument(
        '-d', '--data',
        required=False, type=str, default=None,
        help='Input text as argument (ignored if path is given)')
    parser.add_argument(
        '-o', '--output-folder',
        required=False, dest='output_folder',
        type=str, default='predictions',
        help='Output folder')
    parser.add_argument(
        '-f', '--output-formats',
        required=False, dest='output_formats',
        nargs='+', choices=['csv', 'json'], default=['csv'],
        help='Output folder')
    parser.add_argument(
        '--col',
        required=False, dest='col', type=str, default='text',
        help="In case input is a CSV, use this as the column name. "
             "If the input is a .txt file this option won't have any effect.")
    parser.add_argument(
        '--output-cols',
        required=False, dest='output_cols',
        type=str, default='labels,probabilities,label,probability',
        help="Output columns, provide as comma-separted string. "
             "Only applies to CSV format.")
    parser.add_argument(
        '--no-file-output',
        dest='no_file_output',
        default=False, action='store_true',
        help='Do not write output file '
             '(default: Write output file to `./predictions/` folder)')
    parser.add_argument(
        '--in-parallel',
        default=False, dest='in_parallel', action='store_true',
        help='Run predictions in parallel '
             '(only recommmended for CPU-based models)')
    parser.add_argument(
        '--verbose',
        default=False, dest='verbose', action='store_true',
        help='Print predictions')
    parser.set_defaults(func=lambda args: helpers.predict(**vars(args)))


def generate_config(parser):
    """Generates config for grid search hyperparameter search."""
    parser.add_argument(
        '--name',
        required=True, type=str,
        help='global name prefix and name of output file')
    parser.add_argument(
        '--train-data',
        required=True, dest='train_data', type=str,
        help='train data path')
    parser.add_argument(
        '--test-data',
        required=True, dest='test_data', type=str,
        help='test data path')
    parser.add_argument(
        '-m', '--models',
        required=True, nargs='+',
        help='list of models; '
             'each model will be combined with each param pair')
    parser.add_argument(
        '-p', '--params',
        required=False, nargs='*', default=[],
        help="""
        arbitrary list of grid search params.
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
        '-g', '--globals',
        required=False, dest='global_params', nargs='*', default=[],
        help="list of global params which will be passed "
             "to all runs.\nFormat: 'key:value'")
    parser.set_defaults(
        func=lambda args: helpers.generate_config(**vars(args)))


def generate_text(parser):
    """Generates text."""
    parser.add_argument(
        '--seed',
        required=True, type=str,
        help='Seed text to start generating text from.')

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
        '-n', '--num',
        required=False, dest='n', type=int, default=10,
        help='Number of tweets to generate text from')
    parser.add_argument(
        '--min-tokens',
        required=False, type=int, default=8)
    parser.add_argument(
        '-s', '--source',
        required=False, dest='source', type=str, default='training_data',
        help='Source for seed data (training data (default), or seed_data)')
    parser.add_argument(
        '-r', '--repeats',
        required=False, dest='repeats', type=int, default=1,
        help='Number of repeated times a single seed should be used')
    parser.add_argument(
        '-c', '--contains',
        required=False, dest='should_contain_keyword', type=str, default='',
        help='Only collect sentences which contain keyword')
    parser.add_argument(
        '--n-sentences-after-seed',
        required=False, dest='n_sentences_after_seed', type=int, default=8,
        help='Consider only first n predicted sentences after seed')
    parser.add_argument(
        '--verbose',
        dest='verbose', action='store_true',
        help='Verbose output')
    parser.set_defaults(
        func=lambda args: helpers.augment_training_data(**vars(args)))


def pretrain(parser):
    """Pretrains model based on config.

    Not implemented yet.
    """
    # TODO: implement pretrain


def finetune(parser):
    """Finetunes model based on config.

    The following config keys can/should be present in the config file
    (in runs or params).

    Args:
        name (required): Unique name of the run
        model (required): One of the models which can be finetuned
            (e.g. bert, etc.)
        train_data (required): Path to unannotated data.
            A csv with a text column
            (if only filename is provided it should be located under `data/`)
        test_data: If provided will calculate perplexity
        overwrite: Wipe existing finetuned model with same name
        etc: all additional model-specific parameters
    """
    parser.add_argument(
        '-c', '--config', metavar='C',
        required=False, default='config.json',
        help='Name/path of configuration file. Default: config.json')

    def _finetune(args):
        config_reader = helpers.ConfigReader()
        config = config_reader.parse_fine_tune_config(args.config)
        for run_config in config.runs:
            helpers.finetune(run_config)

    parser.set_defaults(func=_finetune)


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
        required=False, dest='config_path', default='config.json',
        help='Name/path of configuration file. Default: config.json')
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
        required=False, dest='config_path', default='config.json',
        help='Name/path of configuration file. Default: config.json')
    parser.set_defaults(
        func=lambda args: helpers.optimize(**vars(args)))


def ls(parser):
    """List trained models."""
    parser.add_argument(
        '-m', '--model',
        required=False, dest='model', type=str, default=None,
        help='Only show certain models')
    parser.add_argument(
        '-r', '--run-pattern',
        required=False, dest='run_pattern', type=str, default=None,
        help='Filter by run name pattern')
    parser.add_argument(
        '-f', '--filename-pattern',
        required=False, dest='filename_pattern', type=str, default=None,
        help='Filter by name of training data input file')
    parser.add_argument(
        '-p', '--params',
        required=False, type=str, nargs='+', default=None,
        help='Display certain hyperparameters instead of default ones')
    parser.add_argument(
        '-a', '--averaging',
        required=False,
        type=str, choices=['micro', 'macro', 'weighted'], default='macro',
        help='Precision/recall/f1 averaging mode')
    parser.add_argument(
        '-t', '--top',
        required=False, type=int, default=40,
        help='Maximum number of models to show, use -1 to show all')
    parser.add_argument(
        '--metrics',
        required=False, type=str, nargs='+',
        default=['f1', 'precision', 'recall', 'accuracy'],
        choices=['accuracy', 'f1', 'precision', 'recall'],
        help='Metrics to display (also defines sorting order)')
    parser.add_argument(
        '--names-only',
        dest='names_only', action='store_true',
        help='Only list names')
    parser.add_argument(
        '--all-params',
        dest='all_params', action='store_true',
        help='Show all params')
    parser.set_defaults(
        func=lambda args: helpers.helpers.ListRuns().list_runs(**vars(args)))


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
        '-r', '--run',
        required=True, dest='run', type=str,
        help='Name of run')
    parser.add_argument(
        '-p', '--project',
        required=True, dest='project', type=str,
        help='Name of project')
    parser.add_argument(
        '-q', '--question-tag',
        required=True, dest='question_tag', type=str,
        help='Question tag')
    parser.add_argument(
        '-m', '--model_type',
        dest='model_type', type=str, default='fasttext', choices=['fasttext'],
        help='Model type')
    parser.add_argument(
        '-i', '--instance-type',
        dest='instance_type', type=str, default='ml.t2.medium',
        help='Instance type, check https://aws.amazon.com/sagemaker/pricing/instance-types/')
    parser.set_defaults(
        func=lambda args: deploy_helpers.deploy(**vars(args)))
