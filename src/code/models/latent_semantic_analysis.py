from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def latent_query(query: str, documents, doc_ids: list[int]):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}

    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X)

    query_transformed = svd.transform(vectorizer.transform([query]))

    similarities = cosine_similarity(query_transformed, X_reduced)

    sorted_indices = np.argsort(similarities[0])[::-1]

    sorted_documents = [(similarities[0][i], mapping.get(i)) for i in sorted_indices if similarities[0][i] > 0]

    return sorted_documents
