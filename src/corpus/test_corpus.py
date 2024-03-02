from pathlib import Path

from .document import Document
from corpus import Corpus


class TestCorpus(Corpus):
    def __init__(self, path: Path, stemming=False):
        super().__init__(path=path, corpus_type='test', language='english', stemming=stemming)

    def parse_documents(self, path: Path):
        for file in path.glob('*.txt'):
            if not file.is_file():
                continue
            with open(file, 'r') as f:
                text = f.read()
                tokens = self.preprocess_text(text)
                if not tokens:
                    continue
                doc_id = hash(file.stem)
                self.documents.append(Document(doc_id, tokens, file.stem))
