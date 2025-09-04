from typing import Callable

import ir_datasets


def load_rankings(
    filename: str = "data/system_rankings.tsv",
) -> dict[str, list[str]]:
    """Load rankings from file. Every row in the file contains query ID and
    document ID separated by a tab ("\t").

        query_id    doc_id
        646	        4496d63c-8cf5-11e3-833c-33098f9e5267
        646	        ee82230c-f130-11e1-adc6-87dfa8eff430
        646	        ac6f0e3c-1e3c-11e3-94a2-6c66b668ea55

    Example return structure:

    {
        query_id_1: [doc_id_1, doc_id_2, ...],
        query_id_2: [doc_id_1, doc_id_2, ...]
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and list of documents as values.
    """
    rankings = {}

    with open(filename, encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, doc_id = line.strip().split("\t")
            rankings.setdefault(query_id, []).append(doc_id)

    return rankings


def load_ground_truth(
    collection: str = "wapo/v2/trec-core-2018",
) -> dict[str, set[str]]:
    """Load ground truth from ir_datasets. Qrel is a namedtuple class with
    following properties:

        query_id: str
        doc_id: str
        relevance: int
        iteration: str

    relevance is split into levels with values:

        0	not relevant
        1	relevant
        2	highly relevant

    This function considers documents to be relevant for relevance values
        1 and 2.

    Generic structure of returned dictionary:

    {
        query_id_1: {doc_id_1, doc_id_3, ...},
        query_id_2: {doc_id_1, doc_id_5, ...}
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and sets of documents as values.
    """
    dataset = ir_datasets.load(collection)
    ground_truth = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:  # only relevant (1) and highly relevant (2)
            ground_truth.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return ground_truth


def get_precision(
    system_ranking: list[str], ground_truth: set[str], k: int = 100
) -> float:
    """Computes Precision@k.

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k: Cutoff. Only consider system rankings up to k.

    Returns:
        P@K (float).
    """
    if k <= 0:
        return 0.0

    top_k = system_ranking[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in ground_truth)

    return relevant_retrieved / k


def get_average_precision(
    system_ranking: list[str], ground_truth: set[str]
) -> float:
    """Computes Average Precision (AP).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        AP (float).
    """
    if not ground_truth:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(system_ranking, start=1):
        if doc_id in ground_truth:
            relevant_count += 1
            precision_sum += relevant_count / i

    return precision_sum / len(ground_truth)


def get_reciprocal_rank(
    system_ranking: list[str], ground_truth: set[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking, start=1):
        if doc_id in ground_truth:
            return 1.0 / i
    return 0.0


def get_mean_eval_measure(
    system_rankings: dict[str, list[str]],
    ground_truths: dict[str, set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of
            document IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean
            is computed over.

    Returns:
        Mean evaluation measure (float).
    """
    if not system_rankings:
        return 0.0

    scores = []
    for qid, ranking in system_rankings.items():
        if qid not in ground_truths:
            continue
        score = eval_function(ranking, ground_truths[qid])
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    system_rankings = load_rankings()
    ground_truths = load_ground_truth()
