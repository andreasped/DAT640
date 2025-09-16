import ast
import json
from collections import defaultdict
from typing import Callable

import numpy as np
from elasticsearch import Elasticsearch

FIELDS = ["title", "body"]

INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "body": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
        }
    }
}


def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = "toy_index"
) -> list[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = (
            es.search(
                index=index,
                query={"match": {field: t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def load_test_features(filepath: str) -> dict[str, list[list[float]]]:
    """Loads pre-computed feature values from file.

    Args:
        filepath: String (constructed using os.path) of the filepath to a file
        with pre-computed feature values.

    Returns:
        Dictionary with query IDs as keys and a list of query-document feature 
        values as values.
    """
    features = defaultdict(dict)

    with open(filepath, "r") as f:
        for line in f:
            splitline = line.split(";")
            query_id = splitline[0]
            query_doc_features = ast.literal_eval(splitline[1])
            features[query_id] = query_doc_features
    return features


def load_training_data(filepath: str) -> tuple[list[list[float]], list[int]]:
    """Loads pre-computed feature values from file.

    Args:
        filepath: String (constructed using os.path) of the filepath to a file
        with pre-computed feature values.

    Returns:
        Tuple with a list of feature vectors and a list of relevance labels.
    """
    features = []
    labels = []

    # TODO 
    return features, labels


def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    """Indexes documents from JSONL file."""
    bulk_data = []
    with open(filepath, "r") as docs:
        for doc in docs:
            doc = json.loads(doc)
            bulk_data.append({"index": {"_index": index, "_id": doc.pop("doc_id")}})
            bulk_data.append(doc)
    es.bulk(index=index, body=bulk_data, refresh=True)


def reset_index(es: Elasticsearch, index: str) -> None:
    """Reset Index"""
    if es.indices.exists(index):
        es.indices.delete(index=index)

    es.indices.create(index=index, body=INDEX_SETTINGS)


def load_queries(filepath: str) -> dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

    This is an example query:

    ```
    <top>
    <num> Number: OHSU1
    <title> 60 year old menopausal woman without hormone replacement therapy
    <desc> Description:
    Are there adverse effects on lipids when progesterone is given with estrogen replacement therapy
    </top>

    ```

    Take as query ID the value (on the same line) after `<num> Number: `,
    and take as the query string the rest of the line after `<title> `. Omit
    newline characters.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    """
    queries = {}

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("<num> Number:"):
                splitline = line.split(" ", 2)
                query_id = splitline[-1].rstrip()
            if line.startswith("<title>"):
                splitline = line.split(" ", 1)
                queries[query_id] = splitline[-1].rstrip()
    return queries


def load_qrels(filepath: str) -> dict[str, list[str]]:
    """Loads query relevance judgments from a file.
    The qrels file has content with tab-separated values such as the following:

    ```
    MSH1	87056458
    MSH1	87056800
    MSH1	87058606
    MSH2	87049102
    MSH2	87056792
    ```

    Args:
        filepath: String (constructed using os.path) of the filepath to a
            file with queries.

    Returns:
        A dictionary with query IDs and a corresponding list of document IDs
            for documents judged relevant to the query.
    """
    qrels = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            splitline = line.split("\t")
            query_id = splitline[0]
            doc_id = splitline[1].rstrip()
            qrels[query_id].append(doc_id)
    return qrels


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor."""
        # TODO
        self.regressor = ...

    def _train(self, X: list[list[float]], y: list[float]) -> None:
        """Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: list[list[float]], doc_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    query_ids: list[str],
    all_queries: dict[str, str],
    es: Elasticsearch,
    index: str,
    feature_vectors_filepath: str,
    rerank: bool = False,
) -> dict[str, list[str]]:
    """Generate rankings for each of the test query IDs.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        feature_vectors_filepath: String (constructed using os.path) of the
            filepath to a file with pre-computed test feature values.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """

    # Load pre-computed test feature vectors
    # TODO
    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print("Processing query {}/{} ID {}".format(i + 1, len(query_ids), query_id))
        # First-pass retrieval
        query_terms = analyze_query(es, all_queries[query_id], "body", index=index)
        if len(query_terms) == 0:
            print(
                "WARNING: query {} is empty after analysis; ignoring".format(query_id)
            )
            continue
        hits = es.search(index=index, q=" ".join(query_terms), _source=True, size=100)[
            "hits"
        ]["hits"]
        test_rankings[query_id] = [hit["_id"] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            # TODO
            ...
    return test_rankings


def get_reciprocal_rank(system_ranking: list[str], ground_truth: list[str]) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0


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
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, ground_truths[query_id])
    return sum_score / len(system_rankings)


if __name__ == "__main__":
    index_name = "trec9_index"
    es = Elasticsearch(timeout=120)

    reset_index(es, index_name)
    index_documents("data/documents.jsonl", es, index_name)
