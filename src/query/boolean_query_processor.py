from sympy import sympify, to_dnf

from src.utils import to_lower, tokenize
from .query_processor import QueryProcessor


class BooleanQueryProcessor(QueryProcessor):
    def __init__(self, language: str = "english", stemming=False):
        super().__init__(language, stemming)
        self.operators = {'&', '|', '~', '(', ')'}

    def query_to_dnf(self, query):
        clear_query = self.clean_query(query)
        tokens = self.tokenize_boolean_query(clear_query)
        processed_query = ''.join(tokens)
        if processed_query == "":
            return ""
        query_expr = sympify(processed_query, evaluate=False)
        query_dnf = to_dnf(query_expr, simplify=True, force=True)

        return query_dnf

    @staticmethod
    def clean_query(query: str):
        query = to_lower(query)
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        query = " " + query # for resolve when query starts with not
        return query.replace(" not ", " ~ ").replace(" and ", " & ").replace(" or ", " | ")

    def tokenize_boolean_query(self, query: str):
        text = to_lower(query)
        tokens = tokenize(text)

        if self.stemmer is not None:
            tokens = self.stemming(tokens)

        operators = ['&', '|', '~', '(', ')']
        reserved_words = ["pass", "use", "field", "harmonic", "maximum", "print", "input", "variations", "pretty",
                          "test"]

        tokens = [token for token in tokens if token not in reserved_words]

        for i in range(len(tokens) - 1):
            if tokens[i] not in operators and (tokens[i + 1] not in operators or tokens[i + 1] == '~'):
                tokens[i] = tokens[i] + ' &'

        return tokens
