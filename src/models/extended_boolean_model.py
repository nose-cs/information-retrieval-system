from typing import Tuple, List
from sympy import And, Not
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
        is_and = isinstance(dnf_query, And)
        is_negation = isinstance(dnf_query, Not)
        string_dfn_query = str(dnf_query)
        tokens_with_operators = self.query_processor.parse(string_dfn_query, {}, remove_puncts=False)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            if is_and:
                weight_conj_comp, conj_comp_count, _ = self.weight_conjunctive_component(tokens_with_operators, i)
                sim = 1 - (math.sqrt(weight_conj_comp) / math.sqrt(conj_comp_count))
            else:
                weight_dnf, dnf_count = self.weight_disjunctive_normal_form(tokens_with_operators, i)
                sim = math.sqrt(weight_dnf) / math.sqrt(dnf_count)
            if is_negation:
                sim = 1 - sim
            if sim > 0:
                ranking.append((doc.doc_id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def weight_disjunctive_normal_form(self, tokens: List[str], doc_id: int) -> (float, int):
        is_negated = False
        weight_dfn = 0
        dnf_count = 0
        for i in range(len(tokens)):
            if tokens[i] == '(':
                weight_conj_comp, conj_comp_count, i = self.weight_conjunctive_component(tokens, doc_id, i + 1)
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

        return weight_dfn, dnf_count

    def weight_conjunctive_component(self, tokens: List[str], doc_id: int, start: int = 0) -> (float, int, int):
        i = start
        weight_conj_comp = 0
        conj_comp_count = 0
        while i < len(tokens):
            if tokens[i] == ')':
                i += 1
                break
            if tokens[i] == '&':
                i += 1
                continue
            if tokens[i] == '~':
                i += 1
                w_doc = self.weight_doc(tokens[i], doc_id) ** 2
                weight_conj_comp += w_doc
                conj_comp_count += 1
                i += 1
            else:
                w_doc = (1 - self.weight_doc(tokens[i], doc_id)) ** 2
                weight_conj_comp += w_doc
                conj_comp_count += 1
                i += 1
        return weight_conj_comp, conj_comp_count, start

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
