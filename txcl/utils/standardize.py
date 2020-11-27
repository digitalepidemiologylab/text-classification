from .preprocess import (separate_hashtags,
                         standardize_text,
                         anonymize_text,
                         de_emojize)


def standardize_fasttext_pretrain_twitter(text):
    if not isinstance(text, str):
        return ''
    # Separate hashtags
    text = separate_hashtags(text)
    # Standardize text
    text = standardize_text(text)
    # Anonymize
    text = anonymize_text(text)
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text


def standardize_fasttext_twitter(text):
    if not isinstance(text, str):
        return ''
    # demojize
    text = de_emojize(text)
    # separate hashtags
    text = separate_hashtags(text)
    # standardize text
    text = standardize_text(text)
    # anonymize
    text = anonymize_text(text)
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text
