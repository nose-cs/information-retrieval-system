import math
import os

import ir_datasets
import nltk


def remove_punctuation(string: str) -> str:
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\r\x0b\x0c"
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


def idf(corpus: "Corpus", ti: int) -> float:
    N = len(corpus.documents)
    ni = corpus.index.dfs[ti]
    return math.log2(N / ni)


def download_cran_corpus():
    corpus_name = 'cranfield'
    dataset = ir_datasets.load(corpus_name)

    path = f'../../data/corpus/{corpus_name}'
    if os.path.exists(path):
        return print('Corpus already downloaded')

    os.mkdir(path)

    documents = [doc for doc in dataset.docs_iter()]

    documents = documents[:100]

    for i, doc in enumerate(documents):
        with open(f'{path}/{i}.txt', 'w') as f:
            f.write(f'.I {doc.doc_id}\n')
            f.write(f'.T {doc.title}\n')
            f.write(f'.A {doc.author}\n')
            f.write(f'.B {doc.bib}\n')
            f.write(f'.W {doc.text}\n')

    print('Corpus downloaded')


def get_cran_queries():
    corpus_name = 'cranfield'
    dataset = ir_datasets.load(corpus_name)
    queries = [query for query in dataset.queries_iter()]
    return queries[14:18]
