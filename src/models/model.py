"""Module to implement the base method of the IR model"""
from typing import List, Tuple

from corpus import Corpus
from document import Document


class IRModel:
    def __init__(self, doc_analyzer: Corpus):
        self.doc_analyzer = doc_analyzer

    def ranking_function(self, query: List[Tuple[int, int]]) -> List[Tuple[int, float]]:
        """
        Main function that returns a sorted ranking of the similarity
        between the corpus and the query.
        format: [doc_id, similarity]

        :param query: list of tuples (term_id, term_freq)
        :return: list of tuples (doc_id, similarity)
        """
        raise NotImplementedError

    def get_similarity_docs(self, ranking: List[Tuple[int, float]]) -> List[Document]:
        """
        Uses the ranking produced by the ranking function
        and returns the documents with the highest ranking.
        """
        return [self.doc_analyzer.get_document(doc_id) for doc_id, _ in ranking]