from pathlib import Path

import pandas as pd

from corpus import CranCorpus
from evaluation_metrics import precision, recall, f1, fallout, r_precision, r_recall
from models import VectorModel, BooleanModel, ExtendedBooleanModel
from utils import get_cran_queries, download_cran_corpus

download_cran_corpus()
corpus = CranCorpus(Path('../../data/corpus/cranfield'), language='english', stemming=True)
print('Corpus Built')

vector_model = VectorModel(corpus)

boolean_model = BooleanModel(corpus)

extended_boolean_model = ExtendedBooleanModel(corpus)

doc_ids = [str(doc.doc_id) for doc in corpus.documents]
doc_set = set(doc_ids)

queries, qrels = get_cran_queries()

query_ids = [q.query_id for q in queries]

relevant_documents_dict = {}  # dictionary that foreach query_id stores its relevant documents

for qrel in qrels:
    if qrel.relevance < 1:
        continue
    if qrel.query_id not in query_ids:
        continue
    if qrel.doc_id not in doc_ids:
        continue
    if qrel.query_id in relevant_documents_dict:
        relevant_documents_dict[qrel.query_id].append(qrel)
    else:
        relevant_documents_dict[qrel.query_id] = [qrel]

for q_id in relevant_documents_dict:
    relevant_documents_dict[q_id] = sorted(relevant_documents_dict.get(q_id), key=lambda x: x.relevance, reverse=True)

precisions_bool = []
recalls_bool = []
f1s_bool = []
r_precisions_bool = []
r_recalls_bool = []
fallouts_bool = []

precisions = []
recalls = []
f1s = []
r_precisions = []
r_recalls = []
fallouts = []

precisions_vector = []
recalls_vector = []
f1s_vector = []
r_precisions_vector = []
r_recalls_vector = []
fallouts_vector = []

count = 0

for query_to_test in queries:
    query_id = query_to_test.query_id

    try:
        boolean_retrieved_documents = [str(doc.doc_id) for doc in boolean_model.query(query_to_test.text)]
        extended_boolean_retrieved_documents = [str(doc.doc_id) for doc in
                                                extended_boolean_model.query(query_to_test.text)]
        vector_retrieved_documents = [str(doc.doc_id) for doc in vector_model.query(query_to_test.text)]
        relevant_documents = [qrel.doc_id for qrel in relevant_documents_dict.get(query_id)]
        count += 1
    except:
        continue

    r = 5  # r for r-precision and r-recall

    # Calculate metrics for the boolean model
    precisions_bool.append(precision(boolean_retrieved_documents, relevant_documents))
    recalls_bool.append(recall(boolean_retrieved_documents, relevant_documents))
    f1s_bool.append(f1(boolean_retrieved_documents, relevant_documents))
    r_precisions_bool.append(r_precision(boolean_retrieved_documents, relevant_documents, r))
    r_recalls_bool.append(r_recall(boolean_retrieved_documents, relevant_documents, r))
    fallouts_bool.append(fallout(boolean_retrieved_documents, relevant_documents, len(doc_set)))

    # Calculate metrics for the extended boolean model
    precisions.append(precision(extended_boolean_retrieved_documents, relevant_documents))
    recalls.append(recall(extended_boolean_retrieved_documents, relevant_documents))
    f1s.append(f1(extended_boolean_retrieved_documents, relevant_documents))
    r_precisions.append(r_precision(extended_boolean_retrieved_documents, relevant_documents, r))
    r_recalls.append(r_recall(extended_boolean_retrieved_documents, relevant_documents, r))
    fallouts.append(fallout(extended_boolean_retrieved_documents, relevant_documents, len(doc_set)))

    # Calculate metrics for the vector model
    precisions_vector.append(precision(vector_retrieved_documents, relevant_documents))
    recalls_vector.append(recall(vector_retrieved_documents, relevant_documents))
    f1s_vector.append(f1(vector_retrieved_documents, relevant_documents))
    r_precisions_vector.append(r_precision(vector_retrieved_documents, relevant_documents, r))
    r_recalls_vector.append(r_recall(vector_retrieved_documents, relevant_documents, r))
    fallouts_vector.append(fallout(vector_retrieved_documents, relevant_documents, len(doc_set)))

print(f'{count} queries evaluated\n')

data = [
    {
        'Model': 'Boolean',
        'Precision': sum(precisions_bool) / len(precisions_bool),
        'Recall': sum(recalls_bool) / len(recalls_bool),
        'F1': sum(f1s_bool) / len(f1s_bool),
        'R-Precision': sum(r_precisions_bool) / len(r_precisions_bool),
        'R-Recall': sum(r_recalls_bool) / len(r_recalls_bool),
        'Fallout': sum(fallouts_bool) / len(fallouts_bool)
    }, {
        'Model': 'Vector',
        'Precision': sum(precisions_vector) / len(precisions_vector),
        'Recall': sum(recalls_vector) / len(recalls_vector),
        'F1': sum(f1s_vector) / len(f1s_vector),
        'R-Precision': sum(r_precisions_vector) / len(r_precisions_vector),
        'R-Recall': sum(r_recalls_vector) / len(r_recalls_vector),
        'Fallout': sum(fallouts_vector) / len(fallouts_vector)
    },
    {
        'Model': 'Extended',
        'Precision': sum(precisions) / len(precisions),
        'Recall': sum(recalls) / len(recalls),
        'F1': sum(f1s) / len(f1s),
        'R-Precision': sum(r_precisions) / len(r_precisions),
        'R-Recall': sum(r_recalls) / len(r_recalls),
        'Fallout': sum(fallouts) / len(fallouts)
    }
]

metrics_df = pd.DataFrame(data)
print(metrics_df)
