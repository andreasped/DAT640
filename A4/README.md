# Assignment A4: Indexing a document collection

## Task

Build an inverted index using term frequencies for documents in a collection.

## Specific steps

For this assignment, you will need to install additional packages, `requests`, `nltk`, `ir_datasets`, and `sqlitedict`, provided in the `requirements.txt` file.

Data used in the assignment should be downloaded under a `data` folder and the paths should not be changed.

### Collection

In this assignment, we will be using Washington Post collection. This collection is 1.5GB compressed and contains 595,037 documents. It might take a long time to index that many documents on an average computer.

The function to download the collection is provided.

### Preprocessing

The text preprocessing code is provided to ensure that all assignment submissions are consistent in this regard and evaluation can be automated.

### Indexing

Use the two classes provided for representing an inverted index (these should be familiar from the earlier exercises). You'll need to complete and/or create new methods in `InvertedIndex`. The class inherits from `sqlitedict` which can conveniently store key - value pairs on the disk. This is a lightweight addition to sqlite specifically designed to deal with key - value type information. Saving and reading to and from the drive is already implemented automatically for the `self.index` dictionary you need to populate. You have the freedom to choose how you want to construct the keys and values of this dictionary.

The documents are iterated over using [ir_datasets](https://ir-datasets.com/) API. The API returns a NamedTuple containing document ID and several other fields containing text. We will be building a multi-field document index, using the `title` and `body` fields.

There are three methods that you need to complete (i.e., these will be tested):

  * `get_terms(field)` should return all available terms in the field.
  * `get_postings(field, term)` should return a list of postings for the (`field`, `term`) pair.
  * `get_term_frequency(field, term, doc_id)` should return the term frequency for a particular document.

Document texts should be preprocessed using the provided `preprocess()` method.
