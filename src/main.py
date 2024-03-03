from pathlib import Path

from corpus import TestCorpus
from models import VectorModel, BooleanModel
from system import IRSystem

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=True)
model = VectorModel(corpus)
information_retrieval_system = IRSystem(model)

print(information_retrieval_system.query("butterflies"))

query = "butterfield and (B or not kangaroo)"
boolean_model = BooleanModel(corpus)

print(boolean_model.query(query))