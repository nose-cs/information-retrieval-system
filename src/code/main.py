from pathlib import Path

from corpus import TestCorpus, CranCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel
from utils import download_cran_corpus, get_cran_queries


def terminal_main():
    download_cran_corpus()
    corpus = CranCorpus(Path('../../data/corpus/cranfield'),language = 'english', stemming = True)
    print('Corpus Built')

    vector_model = VectorModel(corpus)

    boolean_model = BooleanModel(corpus)

    extended_boolean_model = ExtendedBooleanModel(corpus)

    queries, _ = get_cran_queries()

    for q in queries:
        query = q.text
        print(f'query: {query}')
        # print(f'boolean result: {boolean_model.query(query)}')
        # print(f'vector result: {vector_model.query(query)}')
        ranked_documents = extended_boolean_model.ranking_function(query)
        vector = [doc.doc_id for doc in vector_model.query(query)]
        extended_boolean = [doc[0] for doc in extended_boolean_model.ranking_function1(query)]
        intersect = set(vector) & set(extended_boolean)
        print(f'intersect {len(intersect)}')
        print(f'extended boolean:{len(extended_boolean)}')
        print(f'vector: {len(vector)}')
        # print(f'extended boolean result 1:{ranked_documents}')
        # print(f'extended boolean result 2:{extended_boolean_model.ranking_function1(query)}')
        # extended_boolean_model.user_feedback(query, [4,2], [doc for doc, rank in ranked_documents])
        # extended_boolean_model.pseudo_feedback(query, ranked_documents, 1)
        # recommended_documents = extended_boolean_model.get_recommended_documents()
        # print(f'recommended documents: {recommended_documents}')

def web_main():
    raise NotImplementedError()


if __name__ == '__main__':

    web = False

    if web:
        web_main()
    else:
        terminal_main()
