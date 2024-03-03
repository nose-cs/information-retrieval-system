from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=True)
vector_model = VectorModel(corpus)


query = "butterflies and (butterflies or not kangaroo)"
query_2 = "butterflies are so beautiful that i cry"
boolean_model = BooleanModel(corpus)
extended_boolean_model = ExtendedBooleanModel(corpus)

print(f'boolean: {boolean_model.query(query)}')
print(f'extended boolean: {extended_boolean_model.ranking_function(query_2)}')
print(f'vector: {vector_model.query(query_2)}')