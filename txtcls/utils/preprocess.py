"""
Preprocessing helpers
=====================
"""

import logging
import re
import ast
import html
import unicodedata

import unidecode
import en_core_web_sm
import emoji

from .tokenizer_contractions import CONTRACTIONS

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ImportError:
    logger.warning(
        "Could not import 'bs4', "
        "'txtcls.utils.preprocess.de_emojize' will not work.")

nlp = en_core_web_sm.load()
control_char_regex = re.compile(r'[\r\n\t]+')


def de_emojize(text):
    soup = BeautifulSoup(text, 'html.parser')
    spans = soup.find_all('span')
    if len(spans) == 0:
        return text
    while soup.span is not None:
        emoji_bytes = ast.literal_eval(soup.span.attrs['data-emoji-bytes'])
        emoji_unicode = bytes(emoji_bytes).decode()
        soup.span.replace_with(emoji_unicode)
    return soup.text


def separate_hashtags(text):
    text = re.sub(r"#(\w+)#(\w+)", r" #\1 #\2 ", text)
    return text


def standardize_text(text):
    # Escape HTML symbols
    text = html.unescape(text)
    # Replace \t, \n and \r characters by a whitespace
    text = _remove_control_characters(text)
    # Normalize by compatibility
    text = _normalize(text)
    return text


def _replace_usernames(text, filler='@user'):
    # Note: potentially induces duplicate whitespaces
    username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
    # Replace other user handles by filler
    text = re.sub(username_regex, filler, text)
    # Add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler}')
    return text


def _replace_urls(text, filler='<url>'):
    # Note: includes punctuation in websites
    # Note: potentially induces duplicate whitespaces
    url_regex = re.compile(
        r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
    # Replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # Add spaces between, and remove double spaces again
    text = text.replace(filler, f'{filler}')
    return text


def _replace_email(text, filler='@email'):
    # Note: potentially induces duplicate whitespaces
    email_regex = re.compile(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+')
    # Replace other user handles by filler
    text = re.sub(email_regex, filler, text)
    # Add spaces between, and remove double spaces again
    text = text.replace(filler, f'{filler}')
    return text


def anonymize_text(text, url_filler='<url>',
                   user_filler='@user', email_filler='@email'):
    # Note: potentially induces duplicate whitespaces
    text = _replace_urls(text, filler=url_filler)
    text = _replace_usernames(text, filler=user_filler)
    text = _replace_email(text, filler=email_filler)
    return text


###############################################################################


def _remove_control_characters(text):
    if not isinstance(text, str):
        return text
    # Replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    # Removes all other control characters and the NULL byte
    # (which causes issues when parsing with pandas)
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')


def _expand_contractions(text):
    contractions_pattern = re.compile(
        '({})'.format('|'.join(CONTRACTIONS.keys())),
        flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = \
            CONTRACTIONS.get(match) \
            if CONTRACTIONS.get(match) \
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
    text = ''.join(
        unidecode.unidecode(c)
        if unicodedata.category(c)[0] == 'P' else c for c in text)
    return text


def _remove_punctuation(text):
    """Replaces all symbols of punctuation unicode category except dashes (Pd)"""
    # Note: potentially induces duplicate whitespaces
    text = ''.join(
        ' '
        if unicodedata.category(c)[0] == 'P'
        and unicodedata.category(c)[1] != 'd'
        else c for c in text)
    return text


def _normalize(text):
    """Normalizes unicode strings by compatibilty (in composed form)"""
    return unicodedata.normalize('NFKC', text)


def _remove_emoji(text):
    """Remove all characters of symbols-other (So) unicode category"""
    # Note: potentially induces duplicate whitespaces
    text = ''.join(' ' if unicodedata.category(c) == 'So' else c for c in text)
    return text


def _asciify_emoji(text):
    """Replaces emoji with their descriptions"""
    # Note: potentially induces duplicate whitespaces
    text = emoji.demojize(text)
    # Pad with whitespace
    text = re.sub(r":(\w+):", r" :\1: ", text)
    return text


def _tokenize(text):
    # Create doc
    doc = nlp(text, disable=['parser', 'tagger', 'ner'])
    # Find hashtag indices and merge again (so the # are not lost)
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
