import math
from typing import Tuple, List

from src.code.corpus import Corpus, Document
from src.code.query import BooleanQueryProcessor
from src.code.utils import tf, normalized_idf
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

        # The most similar doc to the query is saved as relevant to the user for the document recommender
        if len(docs) > 0:
            self.document_recommender.add_rating(docs[0].doc_id, 1)

        return docs

    def ranking_function(self, query: str) -> List[Tuple[int, float]]:
        dnf_query = self.query_processor.query_to_dnf(query)
        string_dnf_query = str(dnf_query)
        ranking = []
        for i, doc in enumerate(self.corpus.documents):
            sim = self.get_dnf_weight(string_dnf_query, i)
            if sim > 0:
                ranking.append((doc.doc_id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_dnf_weight(self, dnf: str, doc_id: int) -> float:
        """
        Returns the weight of the disjunctive normal form query for a document,
        in the extended boolean model the or components are calculated with the formula
        sqrt(sum(wi^2) / n) where wi is the weight of the conjunctive component and n is the number
        of conjunctive components

        Args:
        - dnf (str): the disjunctive normal form query
        - doc_id (int): the id of the document
        """
        conjunctive_components = dnf.split(' | ')
        current_is_negated = False
        weight_dnf = 0

        for cc in conjunctive_components:
            if cc[0] == '(' and cc[1] =='~':
                cc = cc[2:]
                current_is_negated = True
            cc_weight = self.get_cc_weight(cc, doc_id)
            if current_is_negated:
                weight_dnf += 1 if cc_weight == 0 else 0
            else:
                weight_dnf += cc_weight ** 2
            current_is_negated = False

        if len(conjunctive_components) == 1:
            return weight_dnf

        return math.sqrt(weight_dnf) / math.sqrt(len(conjunctive_components))

    def get_cc_weight(self, cc: str, doc_id: int) -> float:
        """
        Returns the weight of the conjunctive component for a document,
        in the extended boolean model the and components are calculated with the formula
        1 - sqrt(sum((1 - wi)^2) / n) where wi is the weight of the term and n is the number of terms.

        Args:
        - cc (str): the conjunctive component
        - doc_id (int): the id of the document

        Returns:
        - float: the weight of the conjunctive component for the document
        """
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
            term_weight = self.weight_doc(term, doc_id)
            if current_is_negated:
                w_doc = 0 if term_weight == 0 else 1
            else:
                w_doc = (1 - term_weight) ** 2
            weight_conj_comp += w_doc
            current_is_negated = False

        if len(terms) == 1:
            return 1 - weight_conj_comp

        return 1 - math.sqrt(weight_conj_comp) / math.sqrt(len(terms))

    def weight_doc(self, token: str, dj: int) -> float:
        """
        Returns the weight of a token for a document, it is calculated with the formula
        tf(ti, dj) / max_tf * idf(ti) / max_idf where ti is the token id, tf is the term frequency,
        idf is the inverse document frequency, max_tf is the maximum term frequency and max_idf is the maximum
        inverse document frequency.

        Args:
        - token (str): the token
        - dj (int): the id of the document

        Returns:
        - float: the weight of the token for the document
        """
        try:
            ti = self.corpus.token2id(token)
            return tf(self.corpus, ti, dj) * normalized_idf(self.corpus, ti)
        except KeyError:
            return 0
