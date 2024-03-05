import ir_datasets
import pandas as pd

from evaluation_metrics import precision, recall, f1, fallout, r_precision, r_recall
from models import Query


def terminal_main():
    dataset = ir_datasets.load('cranfield')

    docs = [doc.text for doc in dataset.docs_iter()]

    doc_ids = [doc.doc_id for doc in dataset.docs_iter()]

    queries = [query for query in dataset.queries_iter()]

    qrels = [qrel for qrel in dataset.qrels_iter()]

    relevant_documents_dict = {}  # dictionary that foreach query_id stores its relevant documents

    for qrel in qrels:
        if qrel.relevance < 1:
            continue
        if qrel.query_id in relevant_documents_dict:
            relevant_documents_dict[qrel.query_id].append(qrel)
        else:
            relevant_documents_dict[qrel.query_id] = [qrel]

    for q_id in relevant_documents_dict:
        relevant_documents_dict[q_id] = sorted(relevant_documents_dict.get(q_id), key=lambda x: x.relevance, reverse=True)

    precisions = []
    recalls = []
    f1s_vector = []
    r_precisions_vector = []
    r_recalls_vector = []
    fallouts_vector = []

    count = 0
    for query_to_test in queries:
        query_id = query_to_test.query_id

        latent_semantic_analysis = [doc[1] for doc in Query(query_to_test.text, docs, doc_ids)]
        relevant_documents = [qrel.doc_id for qrel in relevant_documents_dict.get(query_id)]

        k = 5
        precisions.append(precision(latent_semantic_analysis, relevant_documents))
        recalls.append(recall(latent_semantic_analysis, relevant_documents))
        f1s_vector.append(f1(latent_semantic_analysis, relevant_documents))
        r_precisions_vector.append(r_precision(latent_semantic_analysis, relevant_documents, k))
        r_recalls_vector.append(r_recall(latent_semantic_analysis, relevant_documents, k))
        fallouts_vector.append(fallout(latent_semantic_analysis, relevant_documents, len(docs)))

        count += 1
        if count == 24:
            break

    print(f'{count} queries evaluated')
    data = [
        {
            'Model': 'Latent Semantic Analysis',
            'Precision': sum(precisions) / len(precisions),
            'Recall': sum(recalls) / len(recalls),
            'F1': sum(f1s_vector) / len(f1s_vector),
            'R-Precision': sum(r_precisions_vector) / len(r_precisions_vector),
            'R-Recall': sum(r_recalls_vector) / len(r_recalls_vector),
            'Fallout': sum(fallouts_vector) / len(fallouts_vector)
        }
    ]

    metrics_df = pd.DataFrame(data)
    print(metrics_df)

    # download_cran_corpus()
    # corpus = CranCorpus(Path('../../data/corpus/cranfield'),language = 'english', stemming = True)
    # print('Corpus Built')
    #
    # vector_model = VectorModel(corpus)
    #
    # boolean_model = BooleanModel(corpus)
    #
    # extended_boolean_model = ExtendedBooleanModel(corpus)
    #
    # queries, _ = get_cran_queries()
    #
    # for q in queries:
    #     try:
    #         query = q.text
    #         print(f'query: {query}')
    #         # print(f'boolean result: {boolean_model.query(query)}')
    #         # print(f'vector result: {vector_model.query(query)}')
    #         ranked_documents = extended_boolean_model.ranking_function(query)
    #         vector = [doc.doc_id for doc in vector_model.query(query)]
    #
    #         extended_boolean = [doc[0] for doc in extended_boolean_model.ranking_function1(query)]
    #     except:
    #         continue
    #     intersect = set(vector) & set(extended_boolean)
    #     print(f'intersect {len(intersect)}')
    #     print(f'extended boolean:{len(extended_boolean)}')
    #     print(f'vector: {len(vector)}')
    #     # print(f'extended boolean result 1:{ranked_documents}')
    #     # print(f'extended boolean result 2:{extended_boolean_model.ranking_function1(query)}')
    #     # extended_boolean_model.user_feedback(query, [4,2], [doc for doc, rank in ranked_documents])
    #     # extended_boolean_model.pseudo_feedback(query, ranked_documents, 1)
    #     # recommended_documents = extended_boolean_model.get_recommended_documents()
    #     # print(f'recommended documents: {recommended_documents}')


def web_main():
    raise NotImplementedError()


if __name__ == '__main__':

    web = False

    if web:
        web_main()
    else:
        terminal_main()
