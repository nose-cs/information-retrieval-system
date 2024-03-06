import json
from pathlib import Path
from typing import Dict

import numpy as np

from clustering import ClusterManager


class DocumentRecommender:
    """
    Document Recommender class to recommend documents based on the ratings of the user

    Attributes:
    - corpus: the corpus with the documents
    - clusterer: the clusterer used to cluster the documents
    - ratings: the ratings of the documents
    """

    def __init__(self, clusterer: ClusterManager, corpus: "Corpus", ratings: Dict[int, int] = None):
        """
        Document Recommender initialization

        Args:
        - clusterer: corpus with the documents
        - ratings: ratings of the documents consisting in: [doc_id, rating]
        (usually a rating of 0 or 1 if the user found the document interesting)
        """
        self.corpus = corpus
        self.clusterer: ClusterManager = clusterer
        self.name = self.corpus.corpus_type
        if ratings is None:
            self.load_ratings()
        else:
            self.ratings = ratings

    @property
    def mean_of_items(self):
        """Returns the mean of the ratings"""
        return sum(self.ratings.values()) / len(self.ratings)

    def similarity(self, doc_i: int, doc_j: int) -> float:
        """
        Returns the similarity between two documents

        Args:
        - doc_i: the id of the first document
        - doc_j: the id of the second document

        Returns:
        - float: the similarity between the two documents
        """
        return self.jaccard_distance(doc_i, doc_j)

    def jaccard_distance(self, doc_i, doc_j):
        """
        Returns the Jaccard distance between two documents, that is, the intersection over the union of the documents

        Args:
        - doc_i: the id of the first document
        - doc_j: the id of the second document

        Returns:
        - float: the Jaccard distance between the two documents
        """
        vect_i = set(self.corpus.doc2bow(doc_i).keys())
        vect_j = set(self.corpus.doc2bow(doc_j).keys())
        intersect = len(vect_i.intersection(vect_j))
        union = len(vect_i.union(vect_j))
        return intersect / union

    def add_rating(self, doc_id: int, rating: int):
        """
        Adds a rating for the doc_id in the ratings list

        Args:
        - doc_id: the id of the document
        - rating: the rating of the document
        """
        self.ratings[doc_id] = rating
        self.save_ratings()

    def add_ratings(self, ratings: Dict[int, int]):
        """
        Adds the rating for different documents

        Args:
        - ratings: the ratings of the documents consisting in: [doc_id, rating]
        """
        self.ratings.update(ratings)
        self.save_ratings()

    def doc_deviation(self, doc_id: int):
        """
        Mean deviation of a document

        Args:
        - doc_id: the id of the document

        Returns:
        - float: the deviation of the document
        """
        try:
            return self.ratings[doc_id] - self.mean_of_items
        except KeyError:
            # The doc_id is not in the ratings, therefore its value is zero
            return -self.mean_of_items

    def predictor_baseline(self, doc_id: int):
        """
        Returns the predictor baseline of a document, that is the mean of the ratings plus the deviation of the document

        Args:
        - doc_id: the id of the document

        Returns:
        - float: the predictor baseline of the document
        """
        return self.mean_of_items + self.doc_deviation(doc_id)

    def expected_rating(self, doc_id: int):
        """Predicts the rating of an unseen document based on the ratings of the documents that are similar to it.

        Args:
        - doc_id: the id of the document

        Returns:
        - float: the expected rating of the document
        """
        if self.clusterer is not None:
            documents = self.clusterer.get_cluster_samples(doc_id)
        else:
            documents = [doc_id for doc_id in range(len(self.corpus.documents))]
        rated_documents = [doc_id for doc_id in documents if doc_id in self.ratings]
        num = sum(map(lambda d: self.similarity(doc_id, d) * (self.ratings[d] + self.predictor_baseline(d)), rated_documents))
        den = sum(map(lambda d: self.similarity(doc_id, d), rated_documents))
        try:
            return self.predictor_baseline(doc_id) + num / den
        except ZeroDivisionError:
            return 0

    def recommend_documents(self, k=5):
        """
        Searches through all the documents and returns the best `k` recommendations based on the expected rating of the documents.

        Args:
        - k: the number of recommendations to return

        Returns:
        - List[int]: the list of the best `k` recommendations
        """
        doc_ratings = {}
        for doc_id in range(len(self.corpus.documents)):
            if doc_id not in self.ratings:
                predicted_rating = self.expected_rating(doc_id)
                doc_ratings[doc_id] = predicted_rating
        return sorted(doc_ratings, key=lambda x: doc_ratings[x])[:k]

    def load_ratings(self):
        try:
            self.ratings = json.load(open(Path(f'../../data/ratings/{self.name}_ratings.json'), 'r'))
        except FileNotFoundError:
            self.ratings = {}

    def save_ratings(self):
        try:
            json.dump(self.ratings, open(Path(f'../../data/ratings/{self.name}_ratings.json'), 'w'))
        except FileNotFoundError:
            json.dump(self.ratings, open(Path(f'../../data/ratings/{self.name}_ratings.json'), 'x'))
