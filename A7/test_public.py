import random
import time

import A7 as module
import pytest
from elasticsearch.client import Elasticsearch

INDEX_NAME = "trec9_index"


@pytest.fixture(scope="module")
def data_path():
    return "data/{}"


@pytest.fixture(scope="module")
def es():
    return Elasticsearch()


@pytest.fixture(scope="module", autouse=True)
def index_toy_documents(es):
    toy_docs = [
        ("d1", {"title": "t3 t6", "body": "t3 t3 t3 t6 t6"}),
        ("d2", {"title": "t1 t3", "body": "t1 t2 t3 t3 t6"}),
        ("d3", {"title": "t4 t3", "body": "t3 t3 t4 t5"}),
        ("d4", {"title": "t6 t6", "body": "t4 t5 t6 t6"}),
        ("d5", {"title": "t2", "body": "t1 t2 t3 t5"}),
    ]

    if es.indices.exists(index="toy_index"):
        es.indices.delete(index="toy_index")

    es.indices.create(index="toy_index", body=module.INDEX_SETTINGS)
    for doc_id, doc in toy_docs:
        es.index(document=doc, id=doc_id, index="toy_index")
    time.sleep(10)


@pytest.fixture
def toy_query():
    return ["t1 t4", "t2", "t5 t7 t2", "t6 t6"]


def test_indexing(es):
    """This test is 0 points"""
    es.indices.refresh(INDEX_NAME)
    count = es.cat.count(INDEX_NAME, params={"format": "json"})
    assert int(count[0]["count"]) == 54709


def test_query_extraction(es, toy_query):
    """This test is 0.5 points"""
    q0_features = module.extract_query_features(
        module.analyze_query(es, toy_query[0], "body"), es, index="toy_index"
    )
    assert q0_features["query_length"] == 2
    assert q0_features["query_avg_idf"] == pytest.approx(0.9163, abs=1e-4)

    q2_features = module.extract_query_features(
        module.analyze_query(es, toy_query[2], "body"), es, index="toy_index"
    )
    assert q2_features["query_sum_idf"] == pytest.approx(1.4271, abs=1e-4)

    q3_features = module.extract_query_features(
        module.analyze_query(es, toy_query[3], "body"), es, index="toy_index"
    )
    assert q3_features["query_avg_idf"] == pytest.approx(0.5108, abs=1e-4)


def test_document_feature_extraction(es):
    """This test is 0.5 points"""
    d1_features = module.extract_doc_features("d1", es, index="toy_index")
    assert d1_features["doc_length_body"] == 5
    assert d1_features["doc_length_title"] == 2

    d5_features = module.extract_doc_features("d5", es, index="toy_index")
    assert d5_features["doc_length_body"] == 4
    assert d5_features["doc_length_title"] == 1


def test_query_document_feature_extraction(es, toy_query):
    """This test is 0.5 points"""
    q0d2_features = module.extract_query_doc_features(
        module.analyze_query(es, toy_query[0], "body"),
        "d2",
        es,
        index="toy_index",
    )
    assert q0d2_features["unique_query_terms_in_title"] == 1
    assert q0d2_features["sum_TF_body"] == 1

    q0d3_features = module.extract_query_doc_features(
        module.analyze_query(es, toy_query[0], "body"),
        "d3",
        es,
        index="toy_index",
    )
    assert q0d3_features["avg_TF_title"] == 0.5
    assert q0d3_features["max_TF_body"] == 1

    q2d5_features = module.extract_query_doc_features(
        module.analyze_query(es, toy_query[2], "body"),
        "d5",
        es,
        index="toy_index",
    )
    assert q2d5_features["unique_query_terms_in_body"] == 2
    assert q2d5_features["avg_TF_body"] == 1.0


def test_combined_feature_extraction(es, toy_query):
    """This test is 0.5 points"""
    feature_vect_q0_d1 = module.extract_features(
        module.analyze_query(es, toy_query[0], "body"),
        "d1",
        es,
        index="toy_index",
    )
    assert feature_vect_q0_d1 == pytest.approx(
        [
            2,
            1.8325814637483102,
            0.9162907318741551,
            0.9162907318741551,
            2,
            5,
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            0.0,
        ],
        abs=1e-5,
    )

    feature_vect_q1_d2 = module.extract_features(
        module.analyze_query(es, toy_query[1], "body"),
        "d2",
        es,
        index="toy_index",
    )
    assert feature_vect_q1_d2 == pytest.approx(
        [
            1,
            0.9162907318741551,
            0.9162907318741551,
            0.9162907318741551,
            2,
            5,
            0,
            0,
            0,
            0.0,
            1,
            1,
            1,
            1.0,
        ],
        abs=1e-5,
    )

    feature_vect_q3_d3 = module.extract_features(
        module.analyze_query(es, toy_query[3], "body"),
        "d3",
        es,
        index="toy_index",
    )
    assert feature_vect_q3_d3 == pytest.approx(
        [
            2,
            1.0216512475319814,
            0.5108256237659907,
            0.5108256237659907,
            2,
            4,
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            0.0,
        ],
        abs=1e-5,
    )