from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from models.schemas import SemanticDuplicate

_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

_TEXT_SAMPLE_CHARS = 10_000
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            "library_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def find_semantic_duplicates(text: str, arweave_hash: str, top_k: int = 5) -> list[SemanticDuplicate]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    model = _get_model()
    embedding = model.encode([text[:_TEXT_SAMPLE_CHARS]])[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
    )

    duplicates = []
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        if doc_id == arweave_hash:
            continue
        similarity = 1.0 - distance  # cosine distance → similarity
        if similarity >= settings.semantic_similarity_threshold:
            duplicates.append(SemanticDuplicate(arweave_hash=doc_id, similarity=round(similarity, 4)))

    return duplicates


def store_embedding(text: str, arweave_hash: str) -> None:
    collection = _get_collection()
    model = _get_model()
    embedding = model.encode([text[:_TEXT_SAMPLE_CHARS]])[0].tolist()
    collection.upsert(ids=[arweave_hash], embeddings=[embedding])
