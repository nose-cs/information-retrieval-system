"""Corpus module to implement reading and processing of the documents."""

import os
import nltk
from typing import List
from gensim.corpora import Dictionary

from document import Document
from utils import remove_punctuation, to_lower, tokenize


class Corpus:
    def __init__(self, data_dir, language: str = "english", stemming=False):
        self.data_dir = data_dir
        self.documents: List[Document] = []
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.index: Dictionary = None
        if stemming:
            self.stemmer = nltk.PorterStemmer()
        else:
            self.stemmer = None
        try:
            self.load_indexed_document()
        except FileNotFoundError or FileExistsError:
            self.load_data()
            self.create_document_index()
            self.save_indexed_document()
        self.mapping = {doc.doc_id: i for i, doc in enumerate(self.documents)}

    def parse_document(self, doc_id: str, doc_title: str, doc_text: str) -> Document:
        """Parse the document and return a Document object"""
        pass

    def load_data(self):
        """Load the data from the data directory"""
        for file in os.listdir(self.data_dir):
            with open(os.path.join(self.data_dir, file), "r") as f:
                doc_id = file.split(".")[0]
                doc_title = f.name
                doc_text = f.read()
                self.documents.append(self.parse_document(doc_id, doc_title, doc_text))

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess the text and returns a list of tokens"""
        text = remove_punctuation(text)
        text = to_lower(text)
        tokens = tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if self.stemmer is not None:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def create_document_index(self):
        """Create a document index"""
        raise NotImplementedError()

    def load_indexed_document(self):
        """Load the indexed document"""
        raise NotImplementedError()

    def save_indexed_document(self):
        """Save the indexed document"""
        raise NotImplementedError()

    def get_document(self, doc_id: int) -> Document:
        """Get the document by its id"""
        return self.documents[self.mapping[doc_id]]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove the stopwords from the tokens"""
        return [token for token in tokens if token not in self.stopwords]

    def filter_tokens_by_occurrence(self, tokenized_docs, no_below=5, no_above=0.5):
        """Filter the tokens by their occurrence"""
        raise NotImplementedError()

    def build_vocabulary(self, tokenized_docs):
        raise NotImplementedError()
