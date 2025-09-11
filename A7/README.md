# Assignment A7: Feature extraction for learning to rank

## Task

Implement feature extraction for learning-to-rank approach for web search. You are provided with a set of documents, queries, and relevance judgments. Extract query, document and query-document features that can be used in the learning-to-rank approach to retrieval models.

## Background

Machine learning algorithms typically rely on numerical vectors representing the data points that the model trains and tests against. The choices made regarding how to represent text data as numerical vectors have a critical impact on the effectiveness of machine learning for text processing tasks. The process of representing text data as numerical vectors is called *feature extraction*.

You will implement feature-extraction functions to be used for ranking documents.

## Specific steps

For this assignment, you will need to install an additional package, `elasticsearch`, provided in the requirements.txt file.

Data used in the assignment should be downloaded under a `data` folder and the paths should not be changed.

### Elasticsearch

You must have a running local Elasticsearch instance (assuming version 7.17) on your machine. To set up your Elasticsearch instance, follow these steps:

  * Download [Elasticsearch](https://www.elastic.co/downloads/elasticsearch). Choose the correct platform and version 7.17.0.
  * Unzip the downloaded file.
  * Run `bin/elasticsearch`.
  * Check that the server is running by visiting [http://localhost:9200/](http://localhost:9200/).

If needed, review [the Elasticsearch exercise](https://colab.research.google.com/drive/17AFvKgEmrkgzvnn2ZrFkjL1aY2XIVqi5?usp=sharing).

### Test Collections

To help speed up the development phase, you are provided with a toy dataset `toy_index` and the code which indexes this. For the final and hidden tests, a larger dataset (`documents.jsonl`) is used. For both toy dataset used for development and the larger real-world dataset used for the final testing, only two fields will be used: `'title'` and `'body'`. The same `INDEX_SETTINGS` will be used to index both collections.

### Feature extraction

#### Extract Query Features

Implement the function `extract_query_features`, which for a given query and index will consider only those query terms found among the `'body'` fields of the indexed documents. In other words, queries are analyzed with respect to the vocabulary present in the `'body'` field only.

The function should return a dictionary with the keys `'query_length'`, `'query_sum_idf'`, `'query_max_idf'`, and `'query_avg_idf'`. These features, respectively, represent the number of tokens in the analyzed query, as well as the sum, maximum value, and average value of IDF for the same tokens. You will calculate IDF based on the `'body'` field only.

Note: we assume that the query has been analyzed and that query_terms only contain terms that appear in at least one document's body in the collection. It is also assumed that the index always contains a `'body'` field.

#### Extract Document Features

Implement the function `extract_doc_features`, which for a given doc ID and index will quantify features based on the `'title'` and `'body'` fields of the document.

The function should return a dictionary with the keys `'doc_length_title'` and `'doc_length_body'`. These features represent the number of tokens in the title and body, respectively, of the document.

#### Extract Query-Document Features

Implement the function `extract_query_doc_features`, which for a given query, doc ID and index will quantify query-document features based on the `'title'` and `'body'` fields of the document.

The function should return a dictionary with the keys `'unique_query_terms_in_title'`, `'unique_query_terms_in_body'`,`'sum_TF_title'`, `'sum_TF_body'`, `'max_TF_title'`, `'max_TF_body'`, `'avg_TF_title'`, `'avg_TF_body'`.

It should be clear from the structure of how these keys are named which field each should be derived from. Besides the two fields, each query-document feature will be either a count of unique query terms present one of the documents' fields, or else an aggregation function (sum, maximum, or average) over the term frequencies of each query term.

### The TREC'09 Collection

After developing your feature extraction functions, you will now index the `TREC'09` collection, after which you will work with the feature extraction functions to prepare training data for the learning-to-rank approach. For indexing the documents collection, the required function is provided. You will, however, implement functions to parse the queries and relevance judgments from the associated files.

#### Load TREC-9-train queries and relevance judgments

Functions to load the queries and relevance judgments from the corresponding files are provided. More detail about the file structure is in the docstrings.
