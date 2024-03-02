import nltk
import math

def remove_punctuation(string: str) -> str:
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    transform = str.maketrans(punctuation, " " * len(punctuation))
    return string.translate(transform)


def to_lower(string: str):
    return string.lower()


def tokenize(string: str):
    return nltk.wordpunct_tokenize(string)

def tf(corpus: "Corpus", ti: int, dj: int) -> float:
    freq = corpus.get_frequency(ti, dj)
    max_freq_tok, max_freq = corpus.get_max_frequency(dj)
    return freq / max_freq


def idf(doc_analyzer: "Corpus", ti: int) -> float:
    N = len(doc_analyzer.documents)
    ni = doc_analyzer.index.dfs[ti]
    return math.log2(N / ni)