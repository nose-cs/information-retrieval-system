from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel

corpus = TestCorpus(path=Path('../../data/corpus/'), stemming=True)

vector_model = VectorModel(corpus)
boolean_model = BooleanModel(corpus)
extended_boolean_model = ExtendedBooleanModel(corpus)

query_1 = "butterflies and (butterflies or not kangaroo)"
query_2 = "beautiful it aushu butterfield"
query_3 = "not butterflies"

print(f'boolean: {boolean_model.query(query_3)}')
print(f'extended boolean: {extended_boolean_model.ranking_function(query_2)}')
print(f'vector: {vector_model.query(query_2)}')
