from dataclasses import field
import os
import re
import time
from collections import defaultdict
import ir_datasets
import nltk
import requests
from sqlitedict import SqliteDict

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def download_dataset(filename: str, force: bool = False) -> None:
    """Download a dataset to be used with ir_datasets.

    Args:
        filename: Name of the file to download.
        force (optional): Downloads a file and overwrites if already exists.
            Defaults to False.
    """
    filepath = os.path.expanduser(f"~/.ir_datasets/wapo/{filename}")
    if force or not os.path.isfile(filepath):
        print("File downloading...")
        response = requests.get(f"https://gustav1.ux.uis.no/dat640/{filename}")
        if response.ok:
            print("File downloaded; saving to file...")
        with open(filepath, "wb") as f:
            f.write(response.content)

    # This will take some time first time you run it.
    print("First document:\n")
    print(next(ir_datasets.load("wapo/v2/trec-core-2018").docs_iter()))


def preprocess(doc: str) -> list[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]


class InvertedIndex(SqliteDict):
    def __init__(
        self,
        filename: str = "inverted_index.sqlite",
        fields: list[str] = ["title", "body"],
        new: bool = False,
    ) -> None:
        super().__init__(filename, flag="n" if new else "c")
        self.index = {field: defaultdict(list) for field in fields} if new else self

    def add_posting(self, field: str, term: str, doc_id: str, freq: int) -> None:
        """Adds document id and term frequency tuple to the posting list of a 
        term to the appropriate field.

        Args:
            field: Field to which to add the term.
            term: Term for which to add Posting.
            doc_id: Document ID
            freq: Number of times the term appears in the document.
        """
        if not field or not term or not doc_id or freq <= 0:
            return

        self.index[field][term].append((doc_id, freq))

    def add_doc(self, doc: str) -> None:
        """Preprocesses document and adds postings for fields and terms.

        Args:
            doc: Document to index. Expected to be a NamedTuple with attributes
                that correspond to the fields in the InvertedIndex.
        """
        if not hasattr(doc, "doc_id"):
            return

        for field in self.index:
            processed_doc = preprocess(getattr(doc, field) or "")
            if not processed_doc:
                continue  # skip empty content
            term_freq = defaultdict(int)
            for term in processed_doc:
                term_freq[term] += 1
            for term, freq in term_freq.items():
                self.add_posting(field, term, doc.doc_id, freq)

    def get_postings(self, field: str, term: str) -> list[tuple[str, int]]:
        """Fetches the posting list for a given field and term.

        Args:
            field: Field for which to get postings.
            term: Term for which to get postings.

        Returns:
            List of postings for the given term in the given field.
        """
        if not field or not term:
            return []
        field_dict = self.index.get(field)
        if not isinstance(field_dict, dict) and not isinstance(field_dict, defaultdict):
            return []
        return list(field_dict.get(term, []))

    def get_term_frequency(self, field: str, term: str, doc_id: str) -> int:
        """Return the frequency of a given term in a document.

        Args:
            field: Index field.
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        if not field or not term or not doc_id:
            return 0
        postings = self.get_postings(field, term)
        for d, f in postings:
            if d == doc_id:
                return f
        return 0

    def get_terms(self, field: str) -> list[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            List of all terms in a given field.
        """
        if not field:
            return []
        field_dict = self.index.get(field)
        if not isinstance(field_dict, dict) and not isinstance(field_dict, defaultdict):
            return []
        return sorted(field_dict.keys())

    def __exit__(self, *exc_info) -> None:
        if self.flag == "n":
            self.update(self.index)
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)


def index_collection(
    collection: str = "wapo/v2/trec-core-2018",
    filename: str = "inverted_index.sqlite",
    num_documents: int = 595037,
) -> None:
    """Builds an inverted index from a document collection.

    NOTE: WashingtonPost collection has 595037 documents. It might take a very
        long time to index on an average computer.


    Args:
        collection: Collection from ir_datasets.
        filename: Sqlite filename to save index to.
        num_documents: Number of documents to index.
    """
    start = time.time()
    dataset = ir_datasets.load(collection)
    with InvertedIndex(filename, new=True) as index:
        for i, doc in enumerate(dataset.docs_iter()):
            if (i + 1) % (num_documents // 100) == 0:
                print(f"{round(100*(i/num_documents))}% indexed.")
            if i == num_documents:
                break

            index.add_doc(doc)
    print(f"Indexing took {round((time.time()-start)/60)} minutes.")


if __name__ == "__main__":
    download_dataset("WashingtonPost.v2.tar.gz")
    collection = "wapo/v2/trec-core-2018"
    index_file = "inverted_index.sqlite"

    # Comment out this line or delete the index file to re-index.
    if not os.path.isfile(index_file):
        # There are total 595037 documents in the collection.
        # Consider using a smaller subset while developing the index because
        # indexing the entire collection might take a considerable amount of
        # time.
        index_collection(collection, index_file, 50000)

    index = InvertedIndex(index_file)
    print(len(index.get_postings("body", "norway")))
    print(
        index.get_term_frequency("body", "norway", "ebff82c9cd96407d2ef1ba620313f011")
    )
    index.close()
