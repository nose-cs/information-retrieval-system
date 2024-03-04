from sympy import sympify, to_dnf, SympifyError

from src.code.utils import to_lower, tokenize, remove_punctuation
from .query_processor import QueryProcessor


class InvalidQueryException(Exception):
    pass


class BooleanQueryProcessor(QueryProcessor):
    def __init__(self, language: str = "english", stemming=False):
        super().__init__(language, stemming)
        self.operators = {'&', '|', '~', '(', ')'}
        # TODO fill reserved words for handle exceptions
        self.reserved_words = []  # words that raise errors when call the sympify function

    def query_to_dnf(self, query):
        clear_query = self.clean_query(query)
        tokens = self.tokenize_boolean_query(clear_query)
        processed_query = ''.join(tokens)
        if processed_query == "":
            return ""
        try:
            query_expr = sympify(processed_query, evaluate=False)
            query_dnf = to_dnf(query_expr, simplify=True, force=True)
        except TypeError or SympifyError:
            raise InvalidQueryException(f'Invalid query, due to sympify function')
        return query_dnf

    @staticmethod
    def clean_query(query: str):
        query = to_lower(query)
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        query = remove_punctuation(query)
        query = " " + query  # for fix the error parsing when query starts with not
        return query.replace(" not ", " ~ ").replace(" and ", " & ").replace(" or ", " | ")

    def tokenize_boolean_query(self, query: str):
        text = to_lower(query)
        tokens = tokenize(text)

        if self.stemmer is not None:
            tokens = self.stemming(tokens)

        tokens = [token for token in tokens if token not in self.reserved_words]

        for i in range(len(tokens) - 1):
            if tokens[i] not in self.operators and (tokens[i + 1] not in self.operators or tokens[i + 1] == '~'):
                tokens[i] = tokens[i] + ' &'

        return tokens
