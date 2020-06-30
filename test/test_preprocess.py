import pytest

from text_classification.utils.preprocess import (expand_contractions,
                                                  standardize_punctuation,
                                                  remove_emojis,
                                                  asciify_emojis)

def test_expand_contractions():
    text = "weren't isn't aren't"
    assert expand_contractions(text) == 'were not is not are not'

def test_standardize_punctuation():
    text = "‘here’ “are” ´some´ ‵weird‵ ‷punctuations‷; they should be standardized… ¡Olé!"
    text2 = standardize_punctuation(text)
    assert text2 == """'here' "are" ´some´ `weird` ```punctuations```; they should be standardized... !Olé!"""

def test_remove_emojis():
    text = "here are some emojis 😉🤙 that was it"
    text2 = remove_emojis(text)
    assert text2 == """here are some emojis  that was it"""

def test_asciify_emojis():
    text = "here are some emojis 😉🤙 that was it"
    text2 = asciify_emojis(text)
    assert text2 == """here are some emojis :winking_face: :call_me_hand: that was it"""

if __name__ == "__main__":
    pytest.main()
