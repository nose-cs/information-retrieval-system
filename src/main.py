from pathlib import Path

from corpus import TestCorpus

corpus = TestCorpus(path=Path('../data/corpus/'), stemming=False)

print(corpus.documents)