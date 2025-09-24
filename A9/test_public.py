import A9 as module
import pytest
from rank_bm25 import BM25Okapi


@pytest.fixture(scope="module")
def data_path():
    return "data/{}"


@pytest.fixture(scope="module")
def queries(data_path: str):
    return module.load_queries(data_path.format("queries"))


@pytest.fixture(scope="module")
def documents(data_path: str):
    return module.preprocess_documents(data_path.format("documents.jsonl"))


@pytest.fixture
def sample_documents():
    return [
        "Motorcycle safety tips and tricks!",
        "Best practices for motorcycle safe riding",
        "Motorcycle safety guidelines",
        "Advanced motorcycle safety techniques",
    ]


@pytest.fixture
def processed_sample_documents():
    return [
        ["motorcycle", "safety", "tips", "tricks"],
        ["best", "practices", "motorcycle", "safe", "riding"],
        ["motorcycle", "safety", "guidelines"],
        ["advanced", "motorcycle", "safety", "techniques"],
    ]


@pytest.fixture
def sample_query():
    return """
            <num> Number: 1
            <title> motorcycle riding safety tips
        """


def test_doc_preprocess(sample_documents, processed_sample_documents):
    """This test is 0 points"""
    processed = [module.preprocess(doc) for doc in sample_documents]
    assert processed == processed_sample_documents


def test_query_preprocess():
    """This test is 0 points"""
    query = "Motorcycle - Riding safety tips for motorcycle"
    assert module.preprocess(query) == [
        "motorcycle",
        "riding",
        "safety",
        "tips",
        "motorcycle",
    ]


def test_preprocess_documents(data_path):
    """This test is 0 points"""
    doc_file = data_path.format("documents.jsonl")
    documents = module.preprocess_documents(doc_file)
    assert documents[19] == ["academic", "aspirations", "residents"]
    assert len(documents) == 54709


def test_get_top_collection_terms(processed_sample_documents):
    """This test is 0.5 points"""
    prf = module.PRF(prf_num_documents=3, prf_num_terms=3)
    top_ranked_documents = processed_sample_documents[:3]
    top_terms = prf.get_top_collection_terms(top_ranked_documents=top_ranked_documents)
    sorted_top_terms = sorted(top_terms.items(), key=lambda item: item[1], reverse=True)
    assert len(top_terms) == 3
    assert sorted_top_terms[0][0] == "motorcycle"
    assert sorted_top_terms[0][1] == 0.5
    assert sorted_top_terms[1][0] == "safety"
    assert round(sorted_top_terms[1][1], 2) == 0.33


def test_get_query_weighted_terms():
    """This test is 0.5 points"""
    prf = module.PRF()
    query = "motorcycle riding safety tips"
    query_weighted_terms = prf.get_query_weighted_terms(query)
    assert len(query_weighted_terms) == 4
    assert query_weighted_terms["motorcycle"] == 0.25
    assert query_weighted_terms["riding"] == 0.25
    assert query_weighted_terms["safety"] == 0.25
    assert query_weighted_terms["tips"] == 0.25


def test_interpolate_terms():
    """This test is 0.5 points"""
    prf = module.PRF()
    original_terms = {"motorcycle": 0.25, "riding": 0.25, "safety": 0.25, "tips": 0.25}
    new_terms = {"motorcycle": 0.5, "safety": 0.33, "tricks": 0.17}
    interpolated_terms = prf.interpolate_terms(original_terms, new_terms, lam=0.5)

    assert len(interpolated_terms) == 5
    assert interpolated_terms[0][0] == "motorcycle"
    assert interpolated_terms[0][1] == 0.375
    assert interpolated_terms[1][0] == "safety"
    assert round(interpolated_terms[1][1], 2) == 0.29


def test_get_expanded_query(processed_sample_documents):
    """This test is 0.5 points"""
    prf = module.PRF(prf_num_terms=4)
    top_ranked_documents = processed_sample_documents[:3]
    query = "motorcycle riding safety tips"
    expanded_query_terms = prf.get_expanded_query(query, top_ranked_documents)
    assert len(expanded_query_terms) == 5
    assert expanded_query_terms[0][0] == "motorcycle"
    assert expanded_query_terms[1][0] == "safety"