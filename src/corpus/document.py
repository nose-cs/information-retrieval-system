from typing import List

class Document:
    def __init__(self, doc_id, doc_tokens, doc_title="") -> None:
        self.doc_id: int = doc_id
        self.doc_title: str = doc_title
        self.doc_tokens: List[str] = doc_tokens

    def __eq__(self, other):
        return isinstance(other, Document) and self.doc_id == other.doc_id

    def __hash__(self):
        return hash(self.doc_id)

    def __str__(self):
        return f'id: {self.doc_id}, title: {self.doc_title}'

    def to_dict(self):
        return {
            'id': self.doc_id,
            'title': self.doc_title
        }

    def __repr__(self):
        return repr(self.to_dict())
