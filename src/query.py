"""Module to implement the query processing"""
import nltk
from typing import List
from gensim.corpora import Dictionary

from utils import to_lower, remove_punctuation, tokenize


class QueryProcessor:
    def __init__(self, language: str = "english", stemming=False):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        if stemming:
            self.stemmer = nltk.PorterStemmer()
        else:
            self.stemmer = None

    def parse(self, text: str):
        text = to_lower(remove_punctuation(text))
        tokens = tokenize(text)
        tokens = [tok for tok in tokens if tok not in self.stopwords]
        if self.stemmer is not None:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def get_query_vector(self, text: str, index: Dictionary):
        """
        Builds the vector of a query based in the index dictionary
        format: list of (token_id, token_count) 2-tuples.
        """
        return index.doc2bow(self.parse(text))

    def __call__(self, text: str, index: Dictionary):
        return self.get_query_vector(text, index)
