from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _fallback_retrieval(question: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not chunks or not question or len(question.strip()) < 3:
        return []

    texts = [chunk.get("text", "") for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts + [question])
    query_vector = features[-1]
    similarities = cosine_similarity(query_vector, features[:-1]).reshape(-1)
    ranked_indices = similarities.argsort()[::-1][: max(1, top_k)]

    retrieved: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        chunk = chunks[int(idx)].copy()
        chunk["score"] = round(float(similarities[int(idx)]), 4)
        retrieved.append(chunk)
    return retrieved


def retrieve_relevant_chunks(question: str, chunks: List[Dict[str, Any]], embedding_model: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    if not question or not question.strip():
        return []

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover
        return _fallback_retrieval(question, chunks, top_k=top_k)

    try:
        model = SentenceTransformer(embedding_model)
        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if faiss is not None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(np.asarray(embeddings, dtype="float32"))
            scores, indices = index.search(np.asarray(query_embedding, dtype="float32"), min(top_k, len(chunks)))
            retrieved: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                chunk = chunks[int(idx)].copy()
                chunk["score"] = round(float(score), 4)
                retrieved.append(chunk)
            return retrieved
    except Exception:
        return _fallback_retrieval(question, chunks, top_k=top_k)

    return _fallback_retrieval(question, chunks, top_k=top_k)
