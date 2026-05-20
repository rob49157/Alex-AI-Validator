from unittest.mock import MagicMock

import pytest
import numpy as np

import services.semantic_similarity as ss
from models.schemas import SemanticDuplicate


def _unit_vec(seed: int, dims: int = 384) -> list[float]:
    """Deterministic unit vector with a fixed seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dims).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    """Each test gets a fresh in-memory ChromaDB and no cached model."""
    monkeypatch.setattr(ss, "_model", None)
    monkeypatch.setattr(ss, "_client", None)
    monkeypatch.setattr(ss, "_collection", None)
    from config import settings
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path))


@pytest.fixture()
def mock_model(monkeypatch):
    """Inject a mock SentenceTransformer that returns deterministic embeddings."""
    call_count = {"n": 0}

    def fake_encode(texts, **_kwargs):
        call_count["n"] += 1
        return np.array([_unit_vec(call_count["n"])])

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    monkeypatch.setattr(ss, "_model", mock)
    return mock


@pytest.fixture()
def identical_model(monkeypatch):
    """Returns the same embedding for every call — simulates identical content."""
    fixed = np.array([_unit_vec(42)])
    mock = MagicMock()
    mock.encode.return_value = fixed
    monkeypatch.setattr(ss, "_model", mock)
    return mock


def test_empty_collection_returns_no_duplicates(mock_model):
    result = ss.find_semantic_duplicates("any text", "hash-abc")
    assert result == []


def test_store_then_query_different_content(mock_model):
    ss.store_embedding("first book text", "hash-001")
    # Different embedding for the query → low similarity, no duplicate flagged
    duplicates = ss.find_semantic_duplicates("totally different content", "hash-002")
    # With random orthogonal-ish unit vectors, similarity should be below the 0.92 threshold
    assert all(d.similarity < 0.92 for d in duplicates)


def test_identical_content_triggers_duplicate(identical_model):
    from config import settings
    monkeypatch_threshold = 0.90
    original = settings.semantic_similarity_threshold
    settings.semantic_similarity_threshold = monkeypatch_threshold

    try:
        ss.store_embedding("identical book text", "hash-original")
        duplicates = ss.find_semantic_duplicates("identical book text", "hash-copy")
        assert len(duplicates) == 1
        assert duplicates[0].arweave_hash == "hash-original"
        assert duplicates[0].similarity >= monkeypatch_threshold
    finally:
        settings.semantic_similarity_threshold = original


def test_self_not_returned_as_duplicate(identical_model):
    """A doc should never match itself even if it's already in the collection."""
    ss.store_embedding("some text", "hash-self")
    duplicates = ss.find_semantic_duplicates("some text", "hash-self")
    assert all(d.arweave_hash != "hash-self" for d in duplicates)


def test_store_embedding_upserts(mock_model):
    """Calling store_embedding twice with the same hash should not raise."""
    ss.store_embedding("first version", "hash-upsert")
    ss.store_embedding("updated version", "hash-upsert")
    assert ss._get_collection().count() == 1
