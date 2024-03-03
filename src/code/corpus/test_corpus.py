from pathlib import Path

from corpus import Corpus
from .document import Document


class TestCorpus(Corpus):
    def __init__(self, path: Path, stemming=False, language='english'):
        super().__init__(corpus_path=path, corpus_type='test', language=language, stemming=stemming)

    def parse_documents(self, path: Path):
        doc_id = 0
        for file in path.glob('*.txt'):
            if not file.is_file():
                continue
            with open(file, 'r') as f:
                text = f.read()
                tokens = self.preprocess_text(text)
                if not tokens:
                    continue
                doc_id += 1
                self.documents.append(Document(doc_id=doc_id, doc_tokens=tokens, doc_title=file.stem))
