import math
import os
from typing import Dict

import ir_datasets
import nltk


def remove_punctuation(string: str) -> str:
    """Remove punctuation from a string"""
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\r\x0b\x0c"
    transform = str.maketrans(punctuation, " " * len(punctuation))
    return string.translate(transform)


def to_lower(string: str):
    return string.lower()


def tokenize(string: str):
    """Tokenize a string, using wordpunct_tokenize from nltk"""
    return nltk.wordpunct_tokenize(string)


def tf(corpus: "Corpus", ti: int, dj: int) -> float:
    """Returns the normalized term frequency of a term in a document"""
    freq = corpus.get_frequency(ti, dj)
    max_freq_tok, max_freq = corpus.get_max_frequency(dj)

    if max_freq == 0:
        return 0

    return freq / max_freq


def idf(corpus: "Corpus", ti: int) -> float:
    """Returns the inverse document frequency of a term"""
    N = len(corpus.documents)
    ni = corpus.index.dfs[ti]
    return math.log2(N / ni)


def normalized_idf(corpus: "Corpus", ti: int) -> float:
    """Returns the normalized inverse document frequency of a term"""
    N = len(corpus.documents)
    ni = corpus.index.dfs[ti]
    max_idf = corpus.max_idf
    return math.log2(N / ni) / max_idf if max_idf > 0 else 0


def download_cran_corpus_if_not_exist():
    corpus_name = 'cranfield'
    dataset = ir_datasets.load(corpus_name)

    path = f'../../data/corpus/{corpus_name}'
    if os.path.exists(path):
        return print('Corpus already downloaded')

    os.mkdir(path)

    documents = [doc for doc in dataset.docs_iter()]

    # documents = documents[:600]

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
    queries = queries[:100]
    query_ids = [query.query_id for query in queries]
    qrels = [qrel for qrel in dataset.qrels_iter() if qrel.query_id in query_ids]
    return queries, qrels

def get_sorted_relevant_documents_group_by_query(queries, qrels, doc_ids) -> Dict:
    """"
    For each query, find its relevant documents in qrels, if it appears is docs_id

    Args:
        -queries
        -qrels
        -doc_ids

    Return:
        Dict: relevant documents grouped by query_id
    """
    query_ids = [q.query_id for q in queries]

    relevant_documents_dict = {}  # dictionary that foreach query_id stores its relevant documents

    for qrel in qrels:
        if qrel.relevance < 1:
            continue
        if qrel.query_id not in query_ids:
            continue
        if qrel.doc_id not in doc_ids:
            continue
        if qrel.query_id in relevant_documents_dict:
            relevant_documents_dict[qrel.query_id].append(qrel)
        else:
            relevant_documents_dict[qrel.query_id] = [qrel]

    for q_id in relevant_documents_dict:
        relevant_documents_dict[q_id] = sorted(relevant_documents_dict.get(q_id), key=lambda x: x.relevance, reverse=True)

    return relevant_documents_dict
