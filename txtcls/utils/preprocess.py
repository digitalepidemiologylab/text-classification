"""
Preprocessing helpers
=====================
"""

import logging
import re
import html
import unicodedata
import unidecode
from .tokenizer_contractions import CONTRACTIONS
import en_core_web_sm
import emoji
from munch import DefaultMunch


nlp = en_core_web_sm.load()
logger = logging.getLogger(__name__)
control_char_regex = re.compile(r'[\r\n\t]+')


def preprocess(text,
               min_num_tokens=0,
               min_num_chars=0,
               lower_case=False,
               remove_punct=False,
               standardize_punctuation=False,
               asciify=False,
               remove_emojis=False,
               asciify_emojis=False,
               expand_contractions=False,
               lemmatize=False,
               remove_stop_words=False,
               replace_user_with=None,
               replace_url_with=None):
    # 'min_num_tokens': config.get('min_num_tokens', 0),
    # 'min_num_chars': config.get('min_num_chars', 0),
    # 'lower_case': config.get('lower_case', True),
    # 'remove_punct': config.get('remove_punct', False),
    # 'asciify': config.get('asciify', False),
    # 'standardize_punctuation': config.get('standardize_punctuation', True),
    # 'remove_emojis': config.get('remove_emojis', False),
    # 'asciify_emojis': config.get('asciify_emojis', False),
    # 'expand_contractions': config.get('expand_contractions', False),
    # 'lemmatize': config.get('lemmatize', False),
    # 'remove_stop_words': config.get('remove_stop_words', False),
    # 'replace_user_with': config.get('replace_user_with', None),
    # 'replace_url_with': config.get('replace_url_with', None),

    """
    Preprocessing pipeline

    Args:
        min_num_tokens (int): Minimum number of tokens. Default: 0
        min_num_chars (int): Minimum number of character cutoff. Default: 0
        lower_case (bool): Lower case. Default: ``True``
        remove_punct (bool): Remove punctuation. Default: ``False``
        standardize_punctuation (bool): Standardize punctuation. Default: True
        asciify (bool): Asciify accents. Default: ``False``
        remove_emojis (bool): Remove all characters of symbol unicode
            class (S). Default: ``False``
        asciify_emojis (bool): Asciify emojis. Default: ``False``
        expand_contractions (bool): Expand contractions.
            (E.g. `he's` -> `he is`, `wouldn't -> would not`.)
            Note that this may not always be correct.
            Default: ``False``
        lemmatize (bool): Lemmatize strings. Default: ``False``
        remove_stop_words (bool): Remove stop words. Default: ``False``
        replace_user_with (bool): Replace `@user` with something else.
            Default: ``False``
        replace_url_with (bool): Replace `<url>` with something else.
            Default: ``False``

    Returns:
        text (str): Preprocessed text
    """
    # print(min_num_tokens)
    # print(
    #     min_num_tokens, min_num_chars, lower_case, remove_punct,
    #     standardize_punctuation, asciify, remove_emojis, asciify_emojis,
    #     expand_contractions, lemmatize, remove_stop_words, replace_user_with,
    #     replace_url_with)
    text = _remove_control_characters(text)
    # remove HTMl symbols
    text = html.unescape(text)
    # remove accents
    if asciify:
        text = _asciify(text)
    # standardize punctuation
    if standardize_punctuation:
        text = _standardize_punctuation(text)
    # asciify emojis
    if asciify_emojis:
        text = _asciify_emojis(text)
    # remove emojis
    if remove_emojis:
        text = _remove_emojis(text)
    # expand contractions
    if expand_contractions:
        text = _expand_contractions(text)
    # remove user mentions/urls and replace
    if replace_user_with is not None:
        text = text.replace('@user', replace_user_with)
    # replace user/urls with something else
    if replace_url_with is not None:
        text = text.replace('<url>', replace_url_with)
    if min_num_tokens > 0 or remove_punct or lemmatize or remove_stop_words:
        tokens = _tokenize(text)
        # ignore everything below min_num_tokens
        if min_num_tokens > 0:
            num_tokens = sum((
                1 for t in tokens
                if t.is_alpha and
                not t.is_punct and
                t.text.strip()
                not in [replace_user_with, replace_url_with]))
            if num_tokens < min_num_tokens:
                return ''
        # remove punctuation
        if remove_punct:
            tokens = [t for t in tokens if not t.is_punct]
        # remove stop words
        if remove_stop_words:
            tokens = [t for t in tokens if not t.is_stop]
        # merge
        if (remove_stop_words or remove_punct) and not lemmatize:
            text = ' '.join([t.text for t in tokens])
        if lemmatize:
            text = ' '.join([t.lemma_ for t in tokens])
    # lower casing
    if lower_case:
        text = text.lower()
    # min number of character cutoff
    if min_num_chars > 0:
        if len(text) < min_num_chars:
            return ''
    # remove potentially induced duplicate whitespaces
    text = ' '.join(text.split())
    # remove trailing/leading whitespaces
    text = text.strip()
    return text

def get_preprocessing_config(config={}):
    """Generates config file to be used with preprocess() functions and gives default for keys not present in provided config."""
    preprocess_config = DefaultMunch.fromDict({
        'min_num_tokens': config.get('min_num_tokens', 0),
        'min_num_chars': config.get('min_num_chars', 0),
        'lower_case': config.get('lower_case', True),
        'remove_punct': config.get('remove_punct', False),
        'asciify': config.get('asciify', False),
        'standardize_punctuation': config.get('standardize_punctuation', True),
        'remove_emojis': config.get('remove_emojis', False),
        'asciify_emojis': config.get('asciify_emojis', False),
        'expand_contractions': config.get('expand_contractions', False),
        'lemmatize': config.get('lemmatize', False),
        'remove_stop_words': config.get('remove_stop_words', False),
        'replace_user_with': config.get('replace_user_with', None),
        'replace_url_with': config.get('replace_url_with', None),
        }, None)
    return preprocess_config

def _remove_control_characters(s):
    if not isinstance(s, str):
        return s
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def _expand_contractions(text):
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTIONS.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTIONS.get(match)\
                if CONTRACTIONS.get(match)\
                else CONTRACTIONS.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def _asciify(text):
    """Asciify all unicode characters"""
    text = unidecode.unidecode(text)
    return text

def _standardize_punctuation(text):
    text = ''.join(unidecode.unidecode(c) if unicodedata.category(c)[0] == 'P' else c for c in text)
    return text

def _remove_emojis(text):
    """remove all characters of symbol unicode class"""
    text = ''.join('' if unicodedata.category(c)[0] == 'S' else c for c in text)
    return text

def _asciify_emojis(text):
    """remove all characters of symbol unicode class"""
    text = emoji.demojize(text)
    # pad with whitespace
    text = re.sub(r":(\w+):", r" :\1: ", text)
    text = ' '.join(text.split())
    return text

def _tokenize(text):
    # create doc
    doc = nlp(text, disable=['parser', 'tagger', 'ner'])
    # find hashtag indices and merge again (so the # are not lost)
    hashtag_pos = []
    for i, t in enumerate(doc[:-1]):
        if t.text == '#':
            hashtag_pos.append(i)
    with doc.retokenize() as retokenizer:
        for i in hashtag_pos:
            try:
                retokenizer.merge(doc[i:(i+2)])
            except ValueError:
                pass
    return [i for i in doc]
