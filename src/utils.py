import nltk


def remove_punctuation(string: str) -> str:
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    transform = str.maketrans(punctuation, " " * len(punctuation))
    return string.translate(transform)


def to_lower(string: str):
    return string.lower()


def tokenize(string: str):
    return nltk.wordpunct_tokenize(string)
