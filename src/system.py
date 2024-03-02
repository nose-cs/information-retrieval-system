"""Main class that encapsulates all system functionalities"""
from typing import List, Tuple

from query import QueryProcessor
from models import IRModel
from src.corpus.document import Document


class IRSystem:
    def __init__(self, model: IRModel):
        self.model = model
        self.corpus = model.corpus
        stemming = self.corpus.stemmer is not None
        language = self.corpus.language
        self.query_processor = QueryProcessor(language=language, stemming=stemming)

    def query(self, query: str) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        query_vect = self.query_processor(query, self.corpus.index)
        doc_ranking = self.model.ranking_function(query_vect)
        docs = self.model.get_similarity_docs(doc_ranking)
        return docs
