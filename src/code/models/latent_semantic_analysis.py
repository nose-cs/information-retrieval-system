from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = 'impact of renewable energy on environmental sustainability'

documents = [
    "Renewable energy sources such as solar and wind power play a crucial role in combating climate change. They help reduce greenhouse gas emissions and provide a cleaner, more sustainable energy future.",
    "The adoption of renewable energy technologies has been accelerated by global initiatives to address environmental issues. These technologies not only protect the environment but also offer economic benefits by creating jobs in new industries.",
    "While renewable energy is essential for sustainability, its integration into existing power grids presents challenges. Effective storage solutions and infrastructure improvements are necessary to ensure the reliability of energy supply."
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X)

query_transformed = svd.transform(vectorizer.transform([query]))

similarities = cosine_similarity(query_transformed, X_reduced)

sorted_indices = np.argsort(similarities[0])[::-1]

sorted_documents = [(similarities[0][i], i) for i in sorted_indices]
for doc in sorted_documents:
    print(doc)
