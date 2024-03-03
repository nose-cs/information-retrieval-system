from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel, ExtendedBooleanModel
from system import IRSystem

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=True)
model = VectorModel(corpus)
information_retrieval_system = IRSystem(model)

print(information_retrieval_system.query("butterflies"))

query = "butterfield and (butterflies or not kangaroo)"
boolean_model = BooleanModel(corpus)

print(boolean_model.query(query))

extended_boolean_model = ExtendedBooleanModel(corpus)
print(extended_boolean_model.ranking_function(query))