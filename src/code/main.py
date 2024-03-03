from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel


def terminal_main():
    corpus = TestCorpus(path=Path('../../data/corpus/'), stemming=True)
    print('Corpus Built')

    # vector_model = VectorModel(corpus)
    # print('Vector Model Built')
    #
    # boolean_model = BooleanModel(corpus)
    # print('Boolean Model Built')

    extended_boolean_model = ExtendedBooleanModel(corpus)
    print('Extended Boolean Model Built\n')

    query = "beautiful it aushu butterfield are great"

    # print(f'boolean: {boolean_model.query(query)}')
    # print(f'vector: {vector_model.query(query)}')
    ranked_documents = extended_boolean_model.ranking_function(query)
    print(f'extended boolean result:{ranked_documents} \nquery:{query}\n')
    extended_boolean_model.user_feedback(query, [4,2], [doc for doc, rank in ranked_documents])
    extended_boolean_model.pseudo_feedback(query, ranked_documents, 1)
    recommended_documents = extended_boolean_model.get_recommended_documents()
    print(f'recommended documents: {recommended_documents}')

    extended_boolean_model.query(query)


def web_main():
    raise NotImplementedError()


if __name__ == '__main__':

    web = False

    if web:
        web_main()
    else:
        terminal_main()
