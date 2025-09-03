import abc
from collections import Counter
from collections import UserDict as DictClass
from collections import defaultdict
import math
CollectionType = dict[str, dict[str, list[str]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> dict[str, list[str]]:
        """Dictionary of documents for a single field."""
        return {doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc}


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = None,
        fields: list[str] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def score_collection(self, query_terms: list[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        postings = self.index.get(self.field, {}).get(term, [])
        if not postings:
            return

        for doc_id, freq in postings:
            if freq > 0 and query_freq > 0:
                self.scores[doc_id] += freq * query_freq


class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        postings = self.index.get(self.field, {}).get(term, [])
        if not postings:
            return
        nt = len(postings)
        if nt == 0:
            return

        N = len(self.collection)
        if N == 0:
            return
        idf = math.log(N/nt)
        avg_dl = self.collection.avg_field_length(self.field)

        for doc_id, freq in postings:
            dl = len(self.collection[doc_id][self.field])
            denom = freq + self.k1 * (1 - self.b + self.b * abs(dl) / avg_dl)
            if denom <= 0:
                continue

            score = (freq * (1 + self.k1)) / denom
            self.scores[doc_id] += score * idf * query_freq


class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: list[str] = ["title", "body"],
        field_weights: list[float] = [0.2, 0.8],
        bi: list[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        postings_body = self.index.get("body", {}).get(term, [])
        if not postings_body:
            return
        nt = len(postings_body)
        if nt == 0:
            return
        N = len(self.collection)
        if N == 0:
            return
        idf = math.log(N / nt)

        # all docs containing term in any field
        docs_with_term = set()
        for f in self.fields:
            docs_with_term = docs_with_term.union({doc_id for doc_id, _ in self.index.get(f, {}).get(term, [])})

        for doc_id in docs_with_term:
            ctd_tilde = 0.0
            for i, field in enumerate(self.fields):
                postings = dict(self.index.get(field, {}).get(term, []))
                term_freq = postings.get(doc_id, 0)
                if term_freq <= 0:
                    continue

                dl = len(self.collection.get(doc_id, {}).get(field, []))
                avg_field_len = self.collection.avg_field_length(field)
                norm_factor = (1 - self.bi[i]) + self.bi[i] * dl / avg_field_len

                ctd_tilde += self.field_weights[i] * (term_freq / norm_factor)

            if ctd_tilde <= 0:
                continue

            score = (ctd_tilde / (self.k1 + ctd_tilde)) * idf
            self.scores[doc_id] += score * query_freq
