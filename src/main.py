from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel
from system import IRSystem

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=True)
vector_model = VectorModel(corpus)
information_retrieval_system = IRSystem(vector_model)

print(information_retrieval_system.query("butterflies"))

query = "butterflies and (butterflies or not kangaroo)"
query_2 = "butterflies are so beautiful that i cry"
boolean_model = BooleanModel(corpus)

print(boolean_model.query(query))

extended_boolean_model = ExtendedBooleanModel(corpus)
print(extended_boolean_model.ranking_function(query_2))