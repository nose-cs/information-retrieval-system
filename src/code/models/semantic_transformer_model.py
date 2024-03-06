import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

query = 'impact of renewable energy on environmental sustainability'

documents = [
    "Renewable energy sources such as solar and wind power play a crucial role in combating climate change. They help reduce greenhouse gas emissions and provide a cleaner, more sustainable energy future.",
    "The adoption of renewable energy technologies has been accelerated by global initiatives to address environmental issues. These technologies not only protect the environment but also offer economic benefits by creating jobs in new industries.",
    "While renewable energy is essential for sustainability, its integration into existing power grids presents challenges. Effective storage solutions and infrastructure improvements are necessary to ensure the reliability of energy supply."
]

text_total = documents + [query]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(text_total)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

query_tokens = [token.text for token in nlp(query)]

sents_documents = [[sent.text for sent in nlp(doc).sents] for doc in documents]

embeddings_sents_documents = [model.encode(sents_document) for sents_document in sents_documents]

results = [(0, i) for i in range(len(documents))]

for doc_id, embeddings_sents_document in enumerate(embeddings_sents_documents):
    total_relevance = 0

    for token in query_tokens:
        if token in tfidf_feature_names:
            token_idx = tfidf_vectorizer.vocabulary_.get(token)
            tfidf = tfidf_matrix[doc_id, token_idx]

            embedding_token = model.encode(token)
            relevance = util.pytorch_cos_sim(embedding_token, embeddings_sents_document).numpy().flatten().sum()

            relevance_weighted = relevance * tfidf
            total_relevance += relevance_weighted

    results[doc_id] = (total_relevance, doc_id)

results = sorted(results, reverse=True)
print(results)
