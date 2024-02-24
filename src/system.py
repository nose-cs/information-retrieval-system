"""Main class that encapsulates all system functionalities"""
from typing import List, Tuple

from query import QueryProcessor
from models import IRModel
from document import Document


class IRSystem:
    def __init__(self, model: IRModel):
        self.model = model
        self.corpus = model.doc_analyzer
        stemming = self.corpus.stemmer is not None
        self.query_processor = QueryProcessor(stemming)

    def make_query(self, query: str, ranking=False) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        query_vect = self.query_processor(query, self.corpus.index)
        doc_ranking = self.model.ranking_function(query_vect)
        docs = self.model.get_similarity_docs(doc_ranking)
        return docs

    def feedback(self, query: str, relevant_docs: List[int], total_docs: List[int]):
        raise NotImplementedError()

    def pseudo_feedback(self, query: str, ranking: List[Tuple[int, float]], k=10):
        raise NotImplementedError()