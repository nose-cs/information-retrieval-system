from typing import Tuple, List
import math

from .model import IRModel
from src.utils import tf, idf
from src.corpus import Corpus, Document
from src.query import BooleanQueryProcessor


class ExtendedBooleanModel(IRModel):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        stemming = self.corpus.stemmer is not None
        language = self.corpus.language
        self.query_processor = BooleanQueryProcessor(language=language, stemming=stemming)
        self.a = 0.4  # 0.5

    def query(self, query: str) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        doc_ranking = self.ranking_function(query)
        docs = self.get_similarity_docs(doc_ranking)
        return docs

    def ranking_function(self, query: str) -> List[Tuple[int, float]]:
        """
        Main function that returns a sorted ranking of the similarity
        between the corpus and the query.
        format: [doc_id, similarity]
        """
        dnf_query = self.query_processor.query_to_dnf(query)
        string_dfn_query = str(dnf_query)
        tokens_with_operators = self.query_processor.parse(string_dfn_query, {}, remove_puncts=False)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            sim = self.calculate_similarity(tokens_with_operators, i)
            ranking.append((doc.doc_id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def calculate_similarity(self, tokens: List[str], doc_id: int) -> float:
        is_negated = False
        weight_dfn = 0
        dnf_count = 0
        for i in range(len(tokens) - 1):
            if tokens[i] == '(':
                weight_conj_comp = 0
                conj_comp_count = 0
                while tokens[i] != ')':
                    if tokens[i] == '&':
                        i += 1
                        continue
                    if tokens[i] == '~':
                        i += 1
                        w_doc = self.weight_doc(tokens[i], doc_id) ** 2
                        weight_conj_comp += w_doc
                        conj_comp_count += 1
                        continue
                    w_doc = (1 - self.weight_doc(tokens[i], doc_id)) ** 2
                    weight_conj_comp += w_doc
                    conj_comp_count += 1
                    i += 1
                if is_negated:
                    weight_dfn += math.sqrt(weight_conj_comp) / math.sqrt(conj_comp_count)
                    is_negated = False
                else:
                    weight_dfn += 1 - (math.sqrt(weight_conj_comp) / math.sqrt(conj_comp_count))
                dnf_count += 1
            elif tokens[i] == '|':
                continue
            elif tokens[i] == '~':
                is_negated = True
                continue
            else:
                w_doc = self.weight_doc(tokens[i], doc_id) ** 2
                weight_dfn += w_doc
                dnf_count += 1

        sim = math.sqrt(weight_dfn) / math.sqrt(dnf_count)
        return sim

    def weight_doc(self, token: str, dj: int) -> float:
        try:
            ti = self.corpus.token2id(token)
            return self.tf(ti, dj) * self.idf(ti)
        except KeyError:
            return 0

    def tf(self, ti: int, dj: int) -> float:
        return tf(self.corpus, ti, dj)

    def idf(self, ti: int) -> float:
        return idf(self.corpus, ti)
