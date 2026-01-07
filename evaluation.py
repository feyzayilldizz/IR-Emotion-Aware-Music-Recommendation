import numpy as np


def precision_at_k(ranked_docs, relevant_docs, k):
    ranked_docs = ranked_docs[:k]
    rel_set = set(relevant_docs)

    if k == 0:
        return 0.0

    return sum(1 for d in ranked_docs if d in rel_set) / k


def average_precision(ranked_docs, relevant_docs):
    rel_set = set(relevant_docs)
    if not rel_set:
        return 0.0

    score = 0.0
    hits = 0

    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in rel_set:
            hits += 1
            score += hits / i

    return score / len(rel_set)


def mean_average_precision(results, qrels):
    ap_scores = []

    for qid, ranked_docs in results.items():
        relevant_docs = qrels.get(str(qid), [])
        ap_scores.append(average_precision(ranked_docs, relevant_docs))

    return np.mean(ap_scores)


def ndcg_at_k(ranked_docs, relevant_docs, k):
    rel_set = set(relevant_docs)

    def dcg(docs):
        return sum(
            (1.0 / np.log2(i + 2)) if doc in rel_set else 0.0
            for i, doc in enumerate(docs)
        )

    dcg_val = dcg(ranked_docs[:k])

    ideal_docs = list(rel_set)[:k]
    idcg_val = dcg(ideal_docs)

    if idcg_val == 0:
        return 0.0

    return dcg_val / idcg_val
