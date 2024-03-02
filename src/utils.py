import nltk
import math
from sympy import sympify, to_dnf, Not, And, Or

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


def query_to_dnf(query: str):
    tokens = boolean_tokenize(query)
    processed_query = ''.join(tokens)
    # Convert to sympy expression y apply to_dnf
    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True)
    return query_dnf

operators = {'AND': '&', 'NOT': '~', 'OR': '|'}

def boolean_tokenize(query: str):
    tokens = []
    last_is_word = False

    for x in query.split():
        y = x.upper()
        if y in operators:
            tokens.append(operators[y])
            last_is_word = False
        else:
            if last_is_word:
                tokens.append('&')
            tokens.append(x)
            last_is_word = True
    return tokens