from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel


def terminal_main():
    corpus = TestCorpus(path=Path('../../data/corpus/'), stemming=True)
    print('Corpus Built')

    vector_model = VectorModel(corpus)
    print('Vector Model Built')

    boolean_model = BooleanModel(corpus)
    print('Boolean Model Built')

    extended_boolean_model = ExtendedBooleanModel(corpus)
    print('Extended Boolean Model Built')

    query_1 = "butterflies and (butterflies or not kangaroo)"
    query_2 = "beautiful it aushu butterfield"
    query_3 = "not butterflies"

    print(f'boolean: {boolean_model.query(query_3)}')
    print(f'extended boolean: {extended_boolean_model.ranking_function(query_2)}')
    print(f'vector: {vector_model.query(query_2)}')


def web_main():
    raise NotImplementedError()


if __name__ == '__main__':

    web = False

    if web:
        web_main()
    else:
        terminal_main()
