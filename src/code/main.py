from pathlib import Path

from corpus import CranCorpus
from models import ExtendedBooleanModel, IRModel
from utils import download_cran_corpus_if_not_exist
from query import InvalidQueryException


def response_query(query: str, model: IRModel):
    docs = model.query(query)
    print('First 30 results for the query:')
    for doc in docs[:30]:
        print(doc)
    print("Doing pseudo-feedback")
    similarity = model.ranking_function(query)
    model.pseudo_feedback(query, similarity)
    print('Getting recommended documents for you, based on your searches and likes:')
    docs = model.get_recommended_documents()
    for doc in docs:
        if doc is not None:
            print(doc)
    print()


def terminal_main():
    download_cran_corpus_if_not_exist()
    corpus = CranCorpus(Path('../../data/corpus/cranfield'), language='english', stemming=True)
    print('Corpus Built')

    extended_boolean_model = ExtendedBooleanModel(corpus)

    while True:
        print()
        print("Introduce a query")
        query = input()
        try:
            response_query(query, extended_boolean_model)
        except InvalidQueryException:
            print("Your query contains invalid tokens for our sympyfy function :(")
            continue
        except:
            print("Something went wrong :(")
            continue


def web_main():
    raise NotImplementedError()


if __name__ == '__main__':

    web = False

    if web:
        web_main()
    else:
        terminal_main()
