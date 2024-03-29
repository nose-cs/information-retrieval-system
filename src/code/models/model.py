"""Module to implement the base method of the IR model"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.code.corpus import Corpus, Document
from src.code import ClusterManager, DocumentRecommender


class IRModel(ABC):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.clusterer = ClusterManager(self.corpus)
        self.clusterer.fit_cluster(4)
        self.document_recommender = DocumentRecommender(self.clusterer, self.corpus)

    @abstractmethod
    def query(self, query: str) -> List[Document]:
        """
        Main function that returns the documents that are most similar
        to the query.

        Args:
        - query: the query of the user

        Returns:
        - list of documents
        """
        raise NotImplementedError()

    @abstractmethod
    def ranking_function(self, query) -> List[Tuple[int, float]]:
        """
        Returns a sorted ranking of the similarity
        between the corpus and the query.

        Args:
        - query: list of tuples (term_id, term_freq)

        Returns:
        - list of tuples (doc_id, similarity)
        """
        raise NotImplementedError()

    def get_similarity_docs(self, ranking: List[Tuple[int, float]]) -> List[Document]:
        """
        Uses the ranking produced by the ranking function
        and returns the documents with the highest ranking.

        Args:
        - ranking: list of tuples (doc_id, similarity)

        Returns:
        - list of documents
        """
        return [self.corpus.id2doc(doc_id) for doc_id, _ in ranking]

    def user_feedback(self, query: str, relevant_docs: List[int], total_docs: List[int]):
        """
        Feedback if the user helped

        Args:
        -query: The initial query of the user
        -relevant_docs: The list of the documents id the user found relevant
        -total_docs: The total list of documents id that were showed to the user
        """
        non_relevant_docs = [doc_id for doc_id in total_docs if doc_id not in relevant_docs]

        # For the recommender system: relevant docs are stored as interesting, non-relevant as none
        self.document_recommender.add_ratings({doc_id: 1 for doc_id in relevant_docs})

    def pseudo_feedback(self, query: str, ranking: List[Tuple[int, float]], k=10):
        """
        To use if the user didn't help in the feedback.
        The k-highest ranked documents are selected as relevant and the feedback starts.

         Args:
        -query: The initial query of the user
        -ranking: The ranking of the documents
        -k: The number of documents to be considered relevant
        """
        relevant_docs = [doc_id for doc_id, _ in ranking[:k]]
        non_relevant_docs = [doc_id for doc_id, _ in ranking[k:]]

        # For the recommender system: relevant docs are stored as interesting, non-relevant as none
        self.document_recommender.add_ratings({doc_id: 1 for doc_id in relevant_docs})

    def get_recommended_documents(self) -> List[Document]:
        """
        Returns the 5 more interesting unseen documents for the user
        according to the recommendation system
        """
        # If there is no rated document return zero
        if len(self.document_recommender.ratings) == 0:
            return []
        docs_id = self.document_recommender.recommend_documents(5)
        return [self.corpus.documents[doc_id] for doc_id in docs_id]
