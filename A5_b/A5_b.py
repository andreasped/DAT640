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


class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: int) -> None:
        collection_length = self.collection.total_field_length(self.field)
        if collection_length == 0:
            return

        collection_freq = sum(self.collection[doc_id].get(self.field, []).count(term)
                              for doc_id in self.collection)
        p_coll = collection_freq / collection_length

        for doc_id, doc in self.collection.items():
            doc_terms = doc.get(self.field, [])
            dl = len(doc_terms)
            doc_tf = doc_terms.count(term)
            p_td = (1 - self.smoothing_param) * (doc_tf / abs(dl)) + self.smoothing_param * p_coll

            if p_td > 0:
                self.scores[doc_id] += query_freq * math.log(p_td)


class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: list[str] = ["title", "body"],
        field_weights: list[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: float) -> None:
        p_coll_field = {}
        for field in self.fields:
            collection_length = self.collection.total_field_length(field)
            collection_freq = sum(self.collection[doc_id].get(field, []).count(term)
                                  for doc_id in self.collection)
            p_coll_field[field] = collection_freq / max(collection_length, 1)

        for doc_id, doc in self.collection.items():
            p_td = 0.0
            for field, weight in zip(self.fields, self.field_weights):
                doc_terms = doc.get(field, [])
                dl = len(doc_terms)
                doc_tf = doc_terms.count(term)
                p_field = (1 - self.smoothing_param) * (doc_tf / abs(dl)) + \
                          self.smoothing_param * p_coll_field[field]
                p_td += weight * p_field

            if p_td > 0:
                self.scores[doc_id] += query_freq * math.log(p_td)
