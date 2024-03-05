"""Module to implement the evaluation metrics"""
from typing import List


def precision(recovered_documents: List, relevant_documents: List):
    """
    Calculate the accuracy measure, which is the proportion of retrieved documents that are relevant.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.

    Returns:
      double: Value between 0 and 1.
    """
    rr = set(recovered_documents) & set(relevant_documents)

    if len(recovered_documents) == 0:
        return 0

    return len(rr) / len(recovered_documents)


def recall(recovered_documents: List, relevant_documents: List):
    """
    Calculate the recall measure, which is the proportion of relevant documents that were retrieved.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.

    Returns:
      double: Value between 0 and 1.
    """
    return len(set(recovered_documents) & set(relevant_documents)) / len(relevant_documents)


def f_beta(recovered_documents: List, relevant_documents: List, beta):
    """
    Calculate the f measure, which is the weighted harmonic mean of precision and recall.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.
      - beta (double): Weight of the recall measure.

    Returns:
      double: Value between 0 and 1.
    """
    p = precision(recovered_documents, relevant_documents)
    r = recall(recovered_documents, relevant_documents)

    if p == 0 and r == 0:
        return 0

    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def f1(recovered_documents: List, relevant_documents: List):
    """
    Calculate the f1 measure, which is a particular case of the f measure with beta = 1. It is the harmonic mean of precision and recall.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.

    Return:
      double: Value between 0 and 1.
    """
    return f_beta(recovered_documents, relevant_documents, 1)


def fallout(recovered_documents: List, relevant_documents: List, total_documents: int):
    """
    Calculate the fallout measure, which is the proportion of non-relevant documents that are retrieved.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.

    Return:
      double: Value between 0 and 1.
    """
    ri = [d for d in recovered_documents if d not in relevant_documents]
    irrelevant = total_documents - len(relevant_documents)

    if irrelevant == 0:
        return 0

    return len(ri) / irrelevant


def r_precision(recovered_documents: List, relevant_documents: List, r: int):
    """
    Calculate the r-precision measure, which is the precision measure for the first r documents recovered by the SRI.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.
      - r (int): Ranking position to apply the cutoff.

    Return:
      double: Value between 0 and 1.
    """
    return precision(recovered_documents[:r], relevant_documents)


def r_recall(recovered_documents: List, relevant_documents: List, r: int):
    """
    Calculate the r-recall measure, which is the recall measure for the first r documents recovered by the SRI.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.
      - r (int): Ranking position to apply the cutoff.

    Return:
      double: Value between 0 and 1.
    """
    return recall(recovered_documents[:r], relevant_documents)


def r_f1(recovered_documents: List, relevant_documents: List, r: int):
    """
    Calculate the r-f1 measure, which is the f1 measure for the first r documents recovered by the SRI.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.
      - r (int): Ranking position to apply the cutoff.

    Return:
      double: Value between 0 and 1.
    """
    return f1(recovered_documents[:r], relevant_documents)


def r_fallout(recovered_documents: List, relevant_documents: List, total_documents: int, r: int):
    """
    Calculate the r-fallout measure, which is the fallout measure for the first r documents recovered by the SRI.

    Args:
      - recovered_documents (list): Set of documents recovered by the SRI. Each document is defined by its identifier.
      - relevant_documents (list): Set of relevant documents. Each document is defined by its identifier.
      - total_documents (int): Total number of documents in the collection.
      - r (int): Ranking position to apply the cutoff.

    Return:
      double: Value between 0 and 1.
    """
    return fallout(recovered_documents[:r], relevant_documents, total_documents)
