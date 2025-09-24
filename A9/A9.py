"""Classes for query expansion by pseudo-relevance-feedback."""

from collections import Counter
import re
import nltk

import json
from rank_bm25 import BM25Okapi

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

_LAMBDA = 0.5


def load_queries(filepath: str) -> dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

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


def preprocess(text: str) -> list[str]:
    """Pre-processes a string of text. Tokenizes, removes non-alphanumeric
    characters and stopwords.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", text).lower().split()
        if term not in STOPWORDS
    ]


def preprocess_documents(filepath: str) -> list[list[str]]:
    """Loads and preprocesses documents.

    Args:
        filepath: Path to the file containing documents.

    Returns:
        List of preprocessed documents
    """
    documents = []
    with open(filepath, "r") as docs:
        for doc in docs:
            doc_json = json.loads(doc)
            if "title" not in doc_json:
                continue
            else:
                documents.append(preprocess(doc_json["title"]))

    return documents


def get_top_n_documents(
    retriever: BM25Okapi, query: str, tokenized_documents: list[list[str]], n: str
) -> list[list[str]]:
    """Ranks documents using BM25 and return the top_k documents.

    Args:
        retriever: BM25Okapi object.
        query: Query string.
        tokenized_documents: List of tokenized documents.
        n: Number of top documents to return.

    Returns:
        List of top n preprocessed documents.
    """
    return retriever.get_top_n(query.split(), tokenized_documents, n=n)


class PRF:
    def __init__(
        self,
        prf_num_documents: int = 10,
        prf_num_terms: int = 10,
    ) -> None:
        """Pseudo relevance feedback based on RM3 algorithm.

        The algorithm follows
          https://dl.acm.org/doi/pdf/10.1145/3130348.3130376.

        Args:
            prf_num_documents: Number of retrieved documents to use for prf
              (defaults to 10).
            prf_num_terms: Number of top scoring terms to use for prf
              (defaults to 10).
        """
        self.prf_num_documents = prf_num_documents
        self.prf_num_terms = prf_num_terms

    def get_expanded_query(
        self, query: str, top_ranked_documents: list[list[str]]
    ) -> list[tuple[str, float]]:
        """Returns weighted terms to be used for query expansion.

        Args:
            query: Query to use for the initial query retrieval.
            top_ranked_documents: List of top ranked documents

        Returns:
            Dictionary of weighted terms for query expansion.
        """
        query_weighted_terms = self.get_query_weighted_terms(query)
        prf_doc_terms = self.get_top_collection_terms(top_ranked_documents)
        return self.interpolate_terms(query_weighted_terms, prf_doc_terms)

    def interpolate_terms(
        self,
        weighted_terms: dict[str, float],
        weighted_terms_to_add: dict[str, float],
        lam: float = _LAMBDA,
    ) -> list[tuple[str, float]]:
        """Interpolates new weighted terms into the existing query terms.

        Args:
            weighted_terms: Original terms.
            weighted_terms_to_add: Terms to interpolate.
            lam: Weight ratio between old and new terms. If <0.5, new terms will
              be rated higher than old ones.
        """
        # TODO
        return []

    def get_query_weighted_terms(self, query: str) -> dict[str, float]:
        """Returns weighted terms for a given query.

        Args:
            query: Query for the initial retrieval.

        Returns:
            A dictionary with weighted terms.
        """
        # TODO
        return {}

    def get_top_collection_terms(
        self, top_ranked_documents: list[list[str]]
    ) -> dict[str, float]:
        """Returns top terms and weights associated with each term.

        Number of documents to consider and number of terms to take are
        specified in self.num_documents and self.num_terms respectively.

        Args:
            top_ranked_documents: List of top ranked documents.

        Returns:
            A dictionary with weighted terms according to the RM3 algorithm.
        """
        # TODO
        return {}


if __name__ == "__main__":
    documents = preprocess_documents("data/documents.jsonl")
    query = "motorcycle safety tips"

    retriever = BM25Okapi(documents)
    top_ranked_documents = get_top_n_documents(retriever, query, documents, n=10)

    prf = PRF()
    expanded_query_terms = prf.get_expanded_query(query, top_ranked_documents)
    print(expanded_query_terms)
