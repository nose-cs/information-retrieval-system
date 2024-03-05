import re
from pathlib import Path
from typing import List

from corpus import Corpus, Document


# TODO: Add cran documents for more specific searches and recommendations like based on author


class CranCorpus(Corpus):
    """
    The cran corpus has the following structure:
        .I [#] (id of the document)
        .T (title) (can occupy more than one line)
        .A (author) (can occupy more than one line)
        .B (something about the origin)
        .W (the words of the document) (usually they occupy more than one line)
        the first line is the title
    """

    def __init__(self, path: Path, language='english', stemming=False):
        super().__init__(corpus_path=path, corpus_type='cran', language=language, stemming=stemming)

    def parse_documents(self, path: Path):
        for file in path.glob('*.txt'):
            self.parse_document(file)

    def parse_document(self, corpus_path):
        corpus_fd = open(corpus_path, 'r')
        current_id: int = None
        current_lines: List[str] = []
        getting_words = False
        current_title: list = []
        getting_title = False

        for i, line in enumerate(corpus_fd.readlines()):
            m = re.compile(r'\.I (\d+)').match(line)
            if m is not None:
                current_id = int(m.group(1))
                current_lines = []
                getting_words = False
                current_title = []
            if line.startswith('.T'):
                getting_title = True
                current_title.append(line[2:-1])
            elif line.startswith('.W'):
                getting_words = True
                current_lines.append(line[2:-1])
            elif line.startswith('.A'):
                getting_title = False
            elif line.startswith('.X'):
                getting_words = False
            elif getting_words:
                current_lines.append(line[:-1])
            elif getting_title:
                current_title.append(line[:-1])

        if len(current_lines) > 0:
            tokens = self.preprocess_text(" ".join(current_lines))
            title = self.preprocess_text(" ".join(current_title))
            self.documents.append(Document(current_id, tokens, title))
