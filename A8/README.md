# Assignment A8: Learning to rank

## Task

Implement a learning-to-rank approach for web search and evaluate it using a standard test collection. You are provided with a training data and pre-computed features for test queries. Use an initial ranking of top-100 results for each query (based on BM25). Re-rank these documents using a learning-to-rank approach. Specifically, you need to use a pointwise learning-to-rank approach, i.e., any standard regression algorithm that is available in `scikit-learn`.

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

For developing your solutions quickly, you are provided with a toy dataset `toy_index` and the code which indexes this. For the final and hidden tests, you are provided with a larger dataset (`documents.jsonl`). All the functions for indexing the `TREC'09` documents collection, loading the queries and relevance judgments from the corresponding files are provided. For both toy dataset used for development and the larger real-world dataset used for the final testing, only two fields will be used: `'title'` and `'body'`. The same `INDEX_SETTINGS` will be used to index both collections.

#### Pre-computed Feature Vectors

In this assignment you are provided with training feature vectors that you can use for training your learning-to-rank approach. Training data can be found in the `training_data.csv` file. Additionally, you are provided with precomputed feature vectors for the test collection that will be used for evaluating your re-ranking method (`test_features.csv`).

### Training Learning-to-Rank Retrieval Model

Use the provided training feature vectors to train a learning-to-rank model to re-rank the initially retrieved top-k documents (retrieved using Elasticsearch's built-in BM25 implementation) and evaluate against ground truth relevance judgments.

In the training data, for each query we are given feature vectors for the union of all (ground truth) labeled documents and the top-k documents from initial retrieval. We assume retrieved documents without labels are non-relevant.

#### Load the training data

You need to implement the function `load_training_data` that loads pre-computed feature vectors and labels for query and document pairs in the training data.

Each query and document pair have a corresponding label. The label is 1 if a document is relevant to a query, i.e. the document ID is in the QRELS for that query ID. Otherwise, the label is 0. Irrelevant documents are those that appear in top 100 documents retrieved using Elasticsearch.

#### Pointwise-based learning to rank model

You are provided the class for training a learning-to-rank model and ranking documents given feature vectors extracted with respect to a single query. You only need to instantiate a scikit-learn regressor. You have the freedom to choose which one to use.

#### Generate the rankings of documents

Implement the function `get_rankings` to take a trained `PointWiseLTRModel` and the prepared testing data to generate document rankings for each of the queries in a list of test query IDs.

### Evaluation metrics

Use get_reciprocal_rank and get_mean_eval_measure to evaluate the performance of your learning-to-rank model. Once implemented, these functions will allow you to calculate the mean reciprocal rank for both the initial retrieval using BM25 and the re-ranking achieved with the learning-to-rank model. You should observe clear evidence that the learning-to-rank approach outperforms the BM25-based first-pass retrieval, even when applied to a fraction of the available training data.
