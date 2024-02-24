from typing import List

class Document:
    def __init__(self, doc_id: int, doc_tokens: List[str], doc_title: str = "") -> None:
        self.doc_id: int = doc_id
        self.doc_title: str = doc_title
        self.doc_tokens: List[str] = doc_tokens

    def __str__(self):
        return "Document ID: " + str(self.doc_id) + "\nDocument Title: " + self.doc_title

    def __repr__(self):
        return str(self)
