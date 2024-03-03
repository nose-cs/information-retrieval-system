from typing import Tuple, List, Dict

from src.code.corpus import Corpus, Document
from src.code.query import BooleanQueryProcessor
from .model import IRModel


class BooleanModel(IRModel):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        stemming = self.corpus.stemmer is not None
        language = self.corpus.language
        self.query_processor = BooleanQueryProcessor(language=language, stemming=stemming)

    def query(self, query: str) -> List[Document]:
        doc_ranking = self.ranking_function(query)
        docs = self.get_similarity_docs(doc_ranking)
        return docs

    def ranking_function(self, query: str) -> List[Tuple[int, float]]:
        dnf_query = self.query_processor.query_to_dnf(query)
        tokens = self.query_processor.parse(query, self.query_processor.stopwords)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            vector: Dict = {}

            for ti in tokens:
                vector[ti] = self.corpus.get_token_frequency(ti, i) > 0

            sim = dnf_query.subs(vector)

            if sim:
                ranking.append((doc.doc_id, 1))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
