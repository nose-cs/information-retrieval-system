from pathlib import Path

from corpus import TestCorpus
from models import VectorModel
from system import IRSystem

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=True)
model = VectorModel(corpus)
information_retrieval_system = IRSystem(model)

print(information_retrieval_system.query("butterflies"))