import math
from typing import Tuple, List

from sympy import And, Not

from src.code.corpus import Corpus, Document
from src.code.query import BooleanQueryProcessor
from src.code.utils import tf, idf
from .model import IRModel


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

        # Doing clustering to return related documents to the one with the highest score
        if self.clusterer is not None:
            try:
                related_docs = self.clusterer.get_cluster_samples(self.corpus.mapping[docs[0].doc_id])
                related_docs = [self.corpus.documents[doc_id] for doc_id in related_docs[:10]]
                docs = docs[:20] + [d for d in related_docs[:5] if d not in docs[:20]]
            except:
                pass

        # The most similar doc to the query is saved as relevant to the user for the document recommender
        if len(docs) > 0:
            self.document_recommender.add_rating(docs[0].doc_id, 1)

        return docs

    def ranking_function(self, query: str) -> List[Tuple[int, float]]:
        dnf_query = self.query_processor.query_to_dnf(query)
        is_and = isinstance(dnf_query, And)
        is_negation = isinstance(dnf_query, Not)
        string_dfn_query = str(dnf_query)
        tokens_with_operators = self.query_processor.parse(string_dfn_query, {}, remove_puncts=False)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            if is_and:
                weight_conj_comp, conj_comp_count, _ = self.get_cc_weight(tokens_with_operators, i)
                sim = 1 - (math.sqrt(weight_conj_comp) / math.sqrt(conj_comp_count))
            else:
                weight_dnf, dnf_count = self.get_dnf_weight(tokens_with_operators, i)
                sim = math.sqrt(weight_dnf) / math.sqrt(dnf_count)
            if is_negation:
                sim = 1 - sim
            if sim > 0:
                ranking.append((doc.doc_id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_dnf_weight(self, tokens: List[str], doc_id: int) -> (float, int):
        is_negated = False
        weight_dfn = 0
        dnf_count = 0
        for i in range(len(tokens)):
            if tokens[i] == '(':
                weight_conj_comp, conj_comp_count, i = self.get_cc_weight(tokens, doc_id, i + 1)
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

    def get_cc_weight(self, tokens: List[str], doc_id: int, start: int = 0) -> (float, int, int):
        i = start
        weight_conj_comp = 0
        conj_comp_count = 0
        current_is_negated = False
        while i < len(tokens):
            if tokens[i] == ')':
                i += 1
                break
            if tokens[i] == '&':
                i += 1
                continue
            if tokens[i] == '~':
                i += 1
                current_is_negated = True
            else:
                term_weight = self.weight_doc(tokens[i], doc_id)
                if current_is_negated:
                    w_doc = 1 - (1 - term_weight) ** 2
                else:
                    w_doc = (1 - term_weight) ** 2
                weight_conj_comp += w_doc
                conj_comp_count += 1
                current_is_negated = False
                i += 1
        return weight_conj_comp, conj_comp_count, i

    def ranking_function1(self, query: str) -> List[Tuple[int, float]]:
        dnf_query = self.query_processor.query_to_dnf(query)
        string_dfn_query = str(dnf_query)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            sim = self.get_dnf_weight1(string_dfn_query, i)
            if sim > 0:
                ranking.append((doc.doc_id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_dnf_weight1(self, dnf: str, doc_id: int) -> (float, int):
        conjunctive_components = dnf.split(' | ')
        current_is_negated = False
        weight_dnf = 0

        for cc in conjunctive_components:
            if cc[0] == '~':
                cc = cc[1:]
                current_is_negated = True
            cc_weight = self.get_cc_weight1(cc, doc_id)
            if current_is_negated:
                weight_dnf += 1 - cc_weight
            else:
                weight_dnf += cc_weight
            current_is_negated = False

        if len(conjunctive_components) == 1:
            return weight_dnf

        return math.sqrt(weight_dnf) / math.sqrt(len(conjunctive_components))

    def get_cc_weight1(self, cc: str, doc_id: int) -> (float, int):
        terms = cc.split(' & ')
        current_is_negated = False
        weight_conj_comp = 0

        for term in terms:
            if term[0] == '(':
                term = term[1:]
            if terms[-1] == ')':
                term = term[:-1]
            if term[0] == '~':
                term = term[1:]
                current_is_negated = True
            processed_term = self.query_processor.parse(term, {})
            assert len(processed_term) == 1
            term_weight = self.weight_doc(processed_term[0], doc_id)
            if current_is_negated:
                w_doc = 1 - (1 - term_weight) ** 2
            else:
                w_doc = (1 - term_weight) ** 2
            weight_conj_comp += w_doc
            current_is_negated = False

        if len(terms) == 1:
            return 1 - weight_conj_comp

        return 1 - math.sqrt(weight_conj_comp) / math.sqrt(len(terms))

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
