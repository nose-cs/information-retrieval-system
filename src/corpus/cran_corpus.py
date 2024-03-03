import re
from pathlib import Path
from typing import Pattern, List

from corpus import Corpus, Document


# todo: Add cran documents for more specific searches and recommendations like based on author

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
        # Regex to extract the id of the document
        self.id_re: Pattern = re.compile(r'\.I (\d+)')

    def parse_documents(self, corpus_path):
        corpus_fd = open(corpus_path, 'r')
        current_id: int = None
        current_lines: List[str] = []
        getting_words = False
        current_title: list = []
        getting_title = False
        for i, line in enumerate(corpus_fd.readlines()):
            m = self.id_re.match(line)
            if m is not None:
                if len(current_lines) > 0:
                    tokens = self.preprocess_text(" ".join(current_lines))
                    title = self.title_preprocessing(current_title)
                    summary = " ".join(current_lines[:20] + ['...'])
                    self.documents.append(Document(current_id, tokens, title))
                current_id = int(m.group(1))
                current_lines = []
                getting_words = False
                current_title = []
            elif line.startswith('.T'):
                getting_title = True
            elif line.startswith('.W'):
                getting_words = True
            elif line.startswith('.A'):
                getting_title = False
            elif line.startswith('.X'):
                getting_words = False
            elif getting_words:
                current_lines.append(line[:-1])
            elif getting_title:
                current_title.append(line[:-1])

    @staticmethod
    def title_preprocessing(title: List[str]):
        title[0] = title[0].capitalize()
        if title[-1][-1] == '.':
            title[-1] = title[-1][:-1]
        return " ".join(title)
