import random
import time

import A8 as module
import pytest
from elasticsearch.client import Elasticsearch

INDEX_NAME = "trec9_index"


@pytest.fixture(scope="module")
def data_path():
    return "data/{}"


@pytest.fixture(scope="module")
def es():
    return Elasticsearch()


@pytest.fixture(scope="module")
def queries(data_path: str):
    return module.load_queries(data_path.format("queries"))


@pytest.fixture(scope="module")
def qrels(data_path: str):
    return module.load_qrels(data_path.format("qrels"))


@pytest.fixture(scope="module")
def train_test_split(queries):
    random.seed(a=1234567)
    query_ids = sorted(list(queries.keys()))
    random.shuffle(query_ids)
    train_size = int(len(query_ids) * 0.8)
    return query_ids[:train_size], query_ids[train_size:][-100:]


@pytest.fixture(scope="module")
def training_data(data_path):
    # Prepare training data with labels for learning-to-rank
    training_features, labels = module.load_training_data(
        data_path.format("training_data.csv")
    )

    return training_features[:20000], labels[:20000]

@pytest.fixture(scope="module")
def trained_ltr_model(training_data):
    X_train, y_train = training_data
    # Train a model with balanced training data
    positive_labels_count = sum([int(y) for y in y_train])
    negative_training_sample = random.sample(
        [(X, y) for X, y in zip(X_train, y_train) if int(y) == 0], positive_labels_count 
    )
    training_data_balanced = [(X, y) for X, y in zip(X_train, y_train) if int(y) == 1] + negative_training_sample
    X_train_balanced, y_train_balanced = zip(*training_data_balanced)
    # Instantiate PointWiseLTRModel.
    ltr = module.PointWiseLTRModel()
    ltr._train(X_train_balanced, y_train_balanced)
    return ltr

@pytest.fixture(scope="module")
def rankings_ltr(request, es, trained_ltr_model, train_test_split, queries, data_path):
    _, test = train_test_split
    rerank_documents = module.get_rankings(
        trained_ltr_model,
        test,
        queries,
        es,
        index=INDEX_NAME,
        feature_vectors_filepath=data_path.format("test_features.csv"),
        rerank=True,
    )
    yield rerank_documents


def test_indexing(es):
    """This test is 0 points"""
    es.indices.refresh(INDEX_NAME)
    count = es.cat.count(INDEX_NAME, params={"format": "json"})
    assert int(count[0]["count"]) == 54709


def test_load_training_data(data_path):
    """This test is 0.5 points"""
    training_features, labels = module.load_training_data(
        data_path.format("training_data.csv")
    )
    assert len(training_features[100]) == 14
    assert len(labels) == 63751
    assert len(training_features) == 63751
    assert training_features[1000][3] == 5.367367889295858


def test_learning_to_rank_rankings(rankings_ltr):
    """This test is 1 points"""
    assert len(rankings_ltr["MSH2691"]) == 16
    assert rankings_ltr["MSH2691"].index("87254618") < rankings_ltr["MSH2691"].index(
        "87216812"
    )


def test_mean_rr(es, train_test_split, qrels, queries, rankings_ltr, data_path):
    """This test is 0.5 points"""
    _, test = train_test_split

    rankings_first_pass = module.get_rankings(
        None,
        test,
        queries,
        es,
        index=INDEX_NAME,
        feature_vectors_filepath=data_path.format("test_features.csv"),
        rerank=False,
    )

    # Final test: Mean reciprocal rank of LTR and first-pass rankings.

    mrr_first_pass = module.get_mean_eval_measure(
        rankings_first_pass, qrels, module.get_reciprocal_rank
    )
    assert mrr_first_pass > 0.16

    mrr_ltr = module.get_mean_eval_measure(
        rankings_ltr, qrels, module.get_reciprocal_rank
    )
    assert mrr_ltr - mrr_first_pass > 0.005
