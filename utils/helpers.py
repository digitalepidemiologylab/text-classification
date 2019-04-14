from utils.config_reader import ConfigReader
from utils.learning_curve import LearningCurve
from utils.misc import JSONEncoder
from models.bag_of_words import BagOfWordsModel
from models.fasttextmodel import FastTextModel
from models.bertmodel import BERTModel
from models.bert_finetune import BERTFineTune
from models.openai_gpt2 import OpenAIGPT2
from models.dummy_models import DummyModel, RandomModel, WeightedRandomModel
import pandas as pd
import numpy as np
import json
import os
import glob
from multiprocessing import Pool
# import spacy
import string
import unicodedata
import re
from tqdm import tqdm
import csv
from collections import Counter
import sklearn.model_selection
import logging


def train_test_split(test_size=0.2, seed=42, balanced_labels=False):
    """Splits cleaned labelled data into training and test set
    :param test_size: Fraction of data which should be reserved for test data, default: 0.2
    """
    df = get_data()
    logger = logging.getLogger(__name__)
    if balanced_labels:
        df = filter_for_label_balance(df)
    flags = '{}'.format('_balanced' if balanced_labels else '')
    train, test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    for dtype, data in [['train', train], ['test', test]]:
        f_name = '{}_split_{}_seed_{}{}.csv'.format(dtype, int(100*test_size), seed, flags)
        f_path = os.path.join('output', f_name)
        data.to_csv(f_path, index=None, encoding='utf8')
        logger.info('Successfully wrote file {}'.format(f_path))

def get_data(name):
    return pd.read_csv(os.path.join('output', name))

def filter_for_label_balance(df):
    """Performs undersampling for overrepresanted label classes"""
    counts = Counter(df['label'])
    min_count = min(counts.values())
    _df = pd.DataFrame()
    for l in counts.keys():
        _df = pd.concat([_df, df[df['label'] == l].sample(min_count)])
    return _df

def train(run_config):
    model = get_model(run_config.model)
    logger = logging.getLogger(__name__)
    # train
    logger.info('\n\nStart training model for `{}`'.format(run_config.name))
    if not run_config.test_only:
        model.train(run_config)
    # test
    logger.info("\n\nTest results for `{}`:\n".format(run_config.name))
    output = '\n'
    result = model.test(run_config)
    if result is None:
        logger.warning('No test results generated.')
        return
    test_output = os.path.join(run_config.output_path, 'test_output.json')
    if run_config.write_test_output:
        keys = ['text', 'label', 'prediction']
        df = pd.DataFrame({i: result[i] for i in keys})
        df.to_csv(os.path.join(run_config.output_path, 'test_output.csv'))
        for k in keys:
            result.pop(k, None)
    with open(test_output, 'w') as outfile:
        json.dump(result, outfile, cls=JSONEncoder, indent=4)
    print_keys = ['accuracy', 'recall', 'precision', 'f1', 'test_loss', 'loss']
    for key in print_keys:
        for modifier in ['', '_micro', '_macro', '_binary']:
            new_key = key + modifier
            if new_key in result:
                if isinstance(result[new_key], str):
                    output += "{:<20}: {}\n".format(new_key, result[new_key])
                else:
                    output += "{:<20}: {:.4f}\n".format(new_key, result[new_key])
    logger.info(output)
    logger.info("Training for model `{}` finished. Model output written to `{}`".format(run_config.name, run_config.output_path))
        

def read_data(path):
    print('Reading data from {}...'.format(path))
    if path.endswith('.csv'):
        data = []
        with open(path, 'r') as datafile:
            num_lines = sum(1 for line in datafile)
            datafile.seek(0)
            reader = csv.DictReader(datafile)
            for row in tqdm(reader, total=num_lines):
                data.append(row['text'])
    else:
        with open(path, 'r') as datafile:
            num_lines = sum(1 for line in datafile)
            datafile.seek(0)
            data = [line.replace('\n','') for line in tqdm(datafile.readlines(), total=num_lines)]
    return data

def predict(run_name, path=None, data=None, format='csv', output=None, verbose=False):
    config_reader = ConfigReader()
    config_path = os.path.join('output', run_name, 'run_config.json')
    config = config_reader.parse_config(config_path, predict_mode=True)
    logger = logging.getLogger(__name__)
    for run_config in config.runs:
        model = get_model(run_config.model)
        if data is None:
            if path is None:
                raise ValueError('Provide either a path or data argument')
            input_data = read_data(path)
        else:
            input_data = [data]
        logger.info('Predicting...')
        result = model.predict(run_config, data=input_data)
        if data is None:
            if output is None:
                output = os.path.join('output', 'predictions')
            output_file = os.path.join(output, run_config.name + '_' + run_config.model + '.' + format)
            logger.info('Writing output file {}...'.format(output_file))
            with open(output_file, 'w') as outfile:
                if format == 'csv':
                    outfile.write('{},{}\r\n'.format('label', 'probability'))
                    for prediction in result:
                        outfile.write('{},{:f}\r\n'.format(prediction['labels'][0], prediction['probabilities'][0]))
                elif format == 'json':
                    json.dump(result, outfile, cls=JSONEncoder)
        if verbose:
            logger.info('Prediction output:\n')
            logger.info(result)

def get_model(model_name):
    if model_name == 'fasttext':
        return FastTextModel()
    elif model_name == 'bag_of_words':
        return BagOfWordsModel()
    elif model_name == 'bert':
        return BERTModel()
    elif model_name == 'openai_gpt2':
        return OpenAIGPT2()
    elif model_name == 'dummy':
        return DummyModel()
    elif model_name == 'random':
        return RandomModel()
    elif model_name == 'weighted_random':
        return WeightedRandomModel()
    else:
        raise NotImplementedError('Model `{}` is unknown'.format(model_name))

def generate_config(model='fasttext'):
    """Generates a grid search config"""
    ngrams = list(range(1, 4)) 
    epochs = 100 
    dims = [20, 300, 700]
    lr = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]
    runs = []
    config = {
        "params": {
            "train_data": "train_sentiment_split_20_seed_42.csv",
            "test_data": "test_sentiment_split_20_seed_42.csv",
            "overwrite": True
            }
        }
    runs = []
    c = 0
    for ngram in ngrams:
        for d in dims:
            for l in lr: 
                runs.append({'name': 'fasttext{}'.format(c), 'model': 'fasttext', 
                    'dimensions': d, 'num_epochs': epochs, 'ngrams': ngram, 'learning_rate': l}) 
                c += 1
    config['runs'] = runs
    print('Generating config of {} runs...'.format(c))
    with open('config.json', 'w') as f:
        json.dump(config, f)
    print('done')


def collect_results(run='*'):
    """Collects result output for run (by default gets all results)"""
    run_path = os.path.join('output', run)
    folders = glob.glob(run_path)
    results = []
    for f in folders:
        if os.path.isdir(f):
            config_path = os.path.join(f, 'run_config.json')
            test_output_path = os.path.join(f, 'test_output.json')
            try:
                with open(config_path, 'r') as f_p:
                    run_config = json.load(f_p)
                with open(test_output_path, 'r') as f_p:
                    test_output = json.load(f_p)
            except FileNotFoundError:
                continue
            results.append({**run_config, **test_output})
    return pd.DataFrame(results)


def fine_tune(model, run_name, overwrite):
    if model == 'bert':
        finetuner = BERTFineTune(run_name=run_name, overwrite=overwrite)
    else:
        raise NotImplementedError('Model {} is not recognized for fine-tuning'.format(model))
    finetuner.train()


def learning_curve(config_path):
    lc = LearningCurve(config_path)
    lc.init()
    configs = lc.generate_configs()
    for config in configs:
        train(config)


def find_project_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])

def augment_training_data(n=10, min_tokens=8, repeats=1, n_sentences_after_seed=8, source='training_data', should_contain_keyword='', verbose=False):
    raise NotImplementedError
#     # helper function
#     def remove_control_characters(s):
#         return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")
#     # load model
#     model = OpenAIGPT2()
#     try:
#         nlp = spacy.load('en')
#     except:
#         os.system('python -m spacy download en')
#         nlp = spacy.load('en')
#     counts_per_label = int(n/3)
#     counts = {'positive': counts_per_label, 'neutral': counts_per_label, 'negative': counts_per_label}
#     df_augmented = []
#     output_path = os.path.join('data', 'other', 'augmented')
#     if not os.path.isdir(output_path):
#         os.mkdir(output_path)
#     total_collected = 0
#     if source == 'training_data':
#         df_labels = get_cleaned_labelled_data()
#     elif source == 'seed_data':
#         df_labels = pd.read_csv(os.path.join(output_path, 'seed_data.csv'))
#     else:
#         raise NotImplementedError('Source type {} is not supported'.format(source))
#     for k, v in counts.items():
#         num_samples = min(len(df_labels[df_labels['label'] == k]), v)
#         if num_samples == 0:
#             continue
#         sample = df_labels[df_labels['label'] == k].sample(num_samples)
#         pbar = tqdm(sample.iterrows(), total=len(sample))
#         num_collected = 0
#         for i, s in pbar:
#             seed_text = s['text'].replace('@<user>', '').replace('<url>', '')
#             for _ in range(repeats):
#                 pbar.set_description("Num samples collected {:3d}; current class: {}".format(num_collected, k))
#                 gen_text = model.generate_text(seed_text)
#                 sentences = nlp(gen_text).sents
#                 if verbose:
#                     print('Seed text:', seed_text)
#                 for i, sentence in enumerate(sentences):
#                     if i > n_sentences_after_seed:
#                         break
#                     sentence = remove_control_characters(sentence.text)
#                     sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', sentence, flags=re.MULTILINE)
#                     tokenized_sentence = [i.text for i in nlp(sentence) if i.is_alpha]
#                     if len(tokenized_sentence) > min_tokens and should_contain_keyword in sentence.lower():
#                         if verbose:
#                             print(sentence)
#                         df_augmented.append({'text': sentence, 'label': k})
#                         num_collected += 1
#         total_collected += num_collected
#     print("Total samples collected: {}".format(total_collected))
#     if total_collected > 0:
#         df_augmented = pd.DataFrame(df_augmented)
#         if source == 'seed_data':
#             f_name = 'augmented_data_{}_mintokens_{}_repeats_{}_sentences_after_seed_{}.csv'.format(source, min_tokens, repeats, n_sentences_after_seed)
#         else:
#             f_name = 'augmented_data_{}_n_{}_mintokens_{}_repeats_{}_sentences_after_seed_{}.csv'.format(source, n, min_tokens, repeats, n_sentences_after_seed)
#         df_augmented.to_csv(os.path.join(output_path, f_name), index=False)


def generate_fine_tune_input_data(min_tokens=3):
    raise NotImplementedError
#     try:
#         nlp = spacy.load('en')
#     except:
#         os.system('python -m spacy download en')
#         nlp = spacy.load('en')
#     print('Loading data...')
#     df = get_cleaned_data(dtype='anonymized')
#     finetune_data = []
#     print('Generating input data...')
#     for i, tweet in tqdm(enumerate(df['text']), total=len(df)):
#         sentence_was_found = False
#         try:
#             sentences = nlp(tweet).sents
#         except:
#             print('error with sentence:', tweet)
#             continue
#         for sentence in sentences:
#             try:
#                 num_tokens = len([i.text for i in sentence if i.is_alpha])
#             except:
#                 print('error with sentence: "{}" in tweet "{}"'.format(sentence.text, tweet))
#                 break
#             if num_tokens > min_tokens:
#                 sentence_was_found = True
#                 finetune_data.append(sentence.text)
#         if sentence_was_found:
#             # add new line after sentences
#             finetune_data.append('')
#     print('Write output file...')
#     with open(os.path.join(find_folder('other'), 'fine_tuning', 'finetuning.txt'), 'w') as f:
#         for sentence in finetune_data:
#             f.write(sentence + '\n')
