"""Corpus module to implement reading and processing of the documents."""
import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

import nltk
from gensim.corpora import Dictionary

from src.code.utils import remove_punctuation, to_lower, tokenize
from .document import Document


# TODO save tf and idf in data for improve efficiency

class Corpus(ABC):
    """Class to represent a corpus of documents.

    Attributes:
    - corpus_type: string
     Represents the type of the corpus ie. 'cran' or 'test'
    - language: string
    - documents: list of Document objects
    - stopwords: set of string
    - index: gensim.corpora.Dictionary
    It is a mapping from words to their ids and the frequency of the words in the corpus
    - stemmer: nltk.stem.SnowballStemmer
    - vectors: list of dictionaries that represents the bag of words of the documents
    """

    def __init__(self, corpus_path: Path, stemming=False, corpus_type="", language="english"):
        self.corpus_type = corpus_type
        self.language = language
        self.documents: List[Document] = []
        self.stopwords = set(nltk.corpus.stopwords.words(self.language))
        self.index: Dictionary = None
        self.stemmer = nltk.SnowballStemmer(self.language) if stemming else None
        try:
            self.load_indexed_corpus()
        except FileNotFoundError or FileExistsError:
            self.parse_documents(corpus_path)
            self.create_indexed_corpus()
            self.vectors = self.docs2bows()
            self.save_indexed_corpus()
        self.mapping = {doc.doc_id: i for i, doc in enumerate(self.documents)}
        self.max_idf = self._get_max_idf()

    @abstractmethod
    def parse_documents(self, path: Path):
        """Parses the documents from the path and adds them to the documents list."""
        raise NotImplementedError()

    def preprocess_text(self, text: str) -> List[str]:
        """Cleans and tokenizes the text. It also removes the stopwords and stems the tokens if needed."""
        text = remove_punctuation(text)
        text = to_lower(text)
        tokens = tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if self.stemmer is not None:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        """Stems the tokens"""
        return [self.stemmer.stem(tok) for tok in tokens]

    def _get_indexed_corpus_path(self):
        stemmed = '' if self.stemmer is None else '_stemmed'
        return Path(f'../../data/indexed_corpus/{self.corpus_type}{stemmed}/')

    def load_indexed_corpus(self):
        """Loads the indexed corpus from the data folder.

        The indexed corpus is loaded from the folder data/indexed_corpus/corpus_type/ as:
        - index.idx: the index of the words
        - docs.pkl: the documents
        - docs_vect.pkl: the vectors of the documents
        """
        indexed_corpus_path = self._get_indexed_corpus_path()
        self.index = Dictionary.load(f'{indexed_corpus_path}/index.idx')
        self.vectors = pickle.load(open(indexed_corpus_path / 'docs_vect.pkl', 'rb'))
        self.documents = pickle.load(open(indexed_corpus_path / 'docs.pkl', 'rb'))

    def create_indexed_corpus(self):
        """Creates the indexed corpus."""
        docs = [d.doc_tokens for d in self.documents]
        self.index = Dictionary(docs)

    def save_indexed_corpus(self):
        """Saves the indexed corpus in the data folder.

        The indexed corpus is saved in the folder data/indexed_corpus/corpus_type/ as:
        - index.idx: the index of the words
        - docs.pkl: the documents
        - docs_vect.pkl: the vectors of the documents
        """
        indexed_corpus_path = self._get_indexed_corpus_path()
        indexed_corpus_path.mkdir(exist_ok=True)
        self.index.save(f'{indexed_corpus_path}/index.idx')
        pickle.dump(self.vectors, open(indexed_corpus_path / 'docs_vect.pkl', 'wb'))
        pickle.dump(self.documents, open(indexed_corpus_path / 'docs.pkl', 'wb'))

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]

    def id2doc(self, doc_id: int) -> Document:
        """Gets the document matching the id"""
        return self.documents[self.mapping[doc_id]]

    def docs2bows(self) -> List[Dict[int, int]]:
        """
        Converts all the document (the list of words) into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        return [dict(self.index.doc2bow(doc.doc_tokens)) for doc in self.documents]

    def doc2bow(self, doc_id: int) -> Dict[int, int]:
        """
        Converts the document matching the id into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        return self.vectors[doc_id]

    def token2id(self, token: str):
        """Gets the id of a token"""
        return self.index.token2id[token]

    def get_frequency(self, tok_id: int, doc_id: int) -> int:
        """Gets the frequency of a token in certain document"""
        vector = self.doc2bow(doc_id)
        try:
            return vector[tok_id]
        except KeyError:
            return 0

    def get_token_frequency(self, token: str, doc_id: int):
        """Gets the frequency of a token in certain document"""
        try:
            token_id = self.token2id(token)
            return self.get_frequency(token_id, doc_id)
        except KeyError:
            return 0

    def get_max_frequency(self, doc_id: int) -> Tuple[str, int]:
        """Gets the term of the max frequency and its frequency in a certain document"""
        vector = self.doc2bow(doc_id)

        if len(vector) == 0:
            return '', 0

        max_freq_id = max(vector.items(), key=lambda x: x[1])
        return self.index[max_freq_id[0]], max_freq_id[1]

    def _get_max_idf(self):
        """Gets the maximum inverse document frequency of the corpus"""
        N = len(self.documents)
        idfs = [math.log2(N / ni) if ni > 0 else 0 for ni in self.index.dfs]

        if len(idfs) == 0:
            return 0

        return max(idfs)
