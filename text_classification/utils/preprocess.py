import logging
import re
import html
import unicodedata
import unidecode
from text_classification.utils.tokenizer_contractions import CONTRACTIONS
import en_core_web_sm
import emoji
from munch import DefaultMunch


nlp = en_core_web_sm.load()
logger = logging.getLogger(__name__)
control_char_regex = re.compile(r'[\r\n\t]+')

def preprocess(text, config):
    """
    Main function for text preprocessing/standardization.

    Supported config:
    - min_num_tokens: Minimum number of tokens
    - min_num_chars: Minimum number of character cutoff
    - lower_case: Lower case
    - standardize_punctuation: Standardize punctuation
    - asciify: Asciify accents
    - expand_contractions: Expand contractions (such as he's -> he is, wouldn't -> would not, etc. Note that this may not always be correct)
    - lemmatize: Lemmatize strings
    - remove_stop_words: Remove stop words
    - remove_emojis: Remove all characters of symbol unicode class (S)
    - asciify_emojis: Asciify emojis
    - replace_user_with: Replace @user with something else
    - replace_url_with: Replace <url> with something else
    """
    text = remove_control_characters(text)
    # remove HTMl symbols
    text = html.unescape(text)
    # remove accents
    if config.asciify:
        text = asciify(text)
    # standardize punctuation
    if config.standardize_punctuation:
        text = standardize_punctuation(text)
    # asciify emojis
    if config.asciify_emojis:
        text = asciify_emojis(text)
    # remove emojis
    if config.remove_emojis:
        text = remove_emojis(text)
    # expand contractions
    if config.expand_contractions:
        text = expand_contractions(text)
    # remove user mentions/urls and replace
    if config.replace_user_with is not None:
        text = text.replace('@user', config.replace_user_with)
    # replace user/urls with something else
    if config.replace_url_with is not None:
        text = text.replace('<url>', config.replace_url_with)
    if config.min_num_tokens > 0 or config.remove_punct or config.lemmatize or config.remove_stop_words:
        tokens = tokenize(text)
        # ignore everything below min_num_tokens
        if config.min_num_tokens > 0:
            num_tokens = sum((1 for t in tokens if t.is_alpha and not t.is_punct and t.text.strip() not in [config.replace_user_with, config.replace_url_with]))
            if num_tokens < config.min_num_tokens:
                return ''
        # remove punctuation
        if config.remove_punct:
            tokens = [t for t in tokens if not t.is_punct]
        # remove stop words
        if config.remove_stop_words:
            tokens = [t for t in tokens if not t.is_stop]
        # merge
        if (config.remove_stop_words or config.remove_punct) and not config.lemmatize:
            text = ' '.join([t.text for t in tokens])
        if config.lemmatize:
            text = ' '.join([t.lemma_ for t in tokens])
    # lower casing
    if config.lower_case:
        text = text.lower()
    # min number of character cutoff
    if config.min_num_chars > 0:
        if len(text) < config.min_num_chars:
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

def remove_control_characters(s):
    if not isinstance(s, str):
        return s
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def expand_contractions(text):
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

def asciify(text):
    """Asciify all unicode characters"""
    text = unidecode.unidecode(text)
    return text

def standardize_punctuation(text):
    text = ''.join(unidecode.unidecode(c) if unicodedata.category(c)[0] == 'P' else c for c in text)
    return text

def remove_emojis(text):
    """remove all characters of symbol unicode class"""
    text = ''.join('' if unicodedata.category(c)[0] == 'S' else c for c in text)
    return text

def asciify_emojis(text):
    """remove all characters of symbol unicode class"""
    text = emoji.demojize(text)
    # pad with whitespace
    text = re.sub(r":(\w+):", r" :\1: ", text)
    text = ' '.join(text.split())
    return text

def tokenize(text):
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
