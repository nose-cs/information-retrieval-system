from pathlib import Path

import pandas as pd

from corpus import CranCorpus
from evaluation_metrics import precision, recall
from models import VectorModel, BooleanModel, ExtendedBooleanModel
from utils import get_cran_queries

corpus = CranCorpus(Path('../../data/corpus/cranfield'), language='english', stemming=True)
print('Corpus Built')

vector_model = VectorModel(corpus)

boolean_model = BooleanModel(corpus)

extended_boolean_model = ExtendedBooleanModel(corpus)

doc_ids = [doc.doc_id for doc in corpus.documents]
doc_set = set(doc_ids)

queries, qrels = get_cran_queries()

query_ids = [q.query_id for q in queries]

relevant_documents_dict = {}  # dictionary that foreach query_id stores its relevant documents

for qrel in qrels:
    if qrel.doc_id in doc_set:
        continue
    if qrel.query_id in relevant_documents_dict:
        relevant_documents_dict[qrel.query_id].append(qrel)
    else:
        relevant_documents_dict[qrel.query_id] = [qrel]

precisions_bool = []
recalls_bool = []
f1s_bool = []
f3s_bool = []
r_precisions_bool = []

precisions = []
recalls = []

for query_to_test in queries:
    query_id = query_to_test.query_id

    boolean_retrieved_documents = [doc.doc_id for doc in boolean_model.query(query_to_test.text)]
    extended_boolean_retrieved_documents = [doc.doc_id for doc in vector_model.query(query_to_test.text)]
    relevant_documents = [qrel.doc_id for qrel in relevant_documents_dict.get(query_id)]

    # Calculate metrics for the boolean model
    precisions_bool.append(precision(boolean_retrieved_documents, relevant_documents))
    recalls_bool.append(recall(boolean_retrieved_documents, relevant_documents))
    # f1s_bool.append(f1(boolean_retrieved_documents, relevant_documents))

    # Calculate metrics for the extended boolean model
    precisions.append(precision(extended_boolean_retrieved_documents, relevant_documents))
    recalls.append(recall(extended_boolean_retrieved_documents, relevant_documents))
    # f1s_bool.append(f1(extended_boolean_retrieved_documents, relevant_documents))

data = [
    {
        'Model': 'Boolean',
        'Precision': sum(precisions_bool) / len(precisions_bool),
        'Recall': sum(recalls_bool) / len(recalls_bool)
    },
    {
        'Model': 'Extended',
        'Precision': sum(precisions) / len(precisions),
        'Recall': sum(recalls) / len(recalls),
    }
]

metrics_df = pd.DataFrame(data)
print(metrics_df)
