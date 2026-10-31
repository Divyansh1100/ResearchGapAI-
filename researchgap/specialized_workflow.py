from __future__ import annotations

import time
from typing import Any, Dict, List

from .general_llm import call_chat_completion
from .text_retrieval import retrieve_relevant_chunks


def format_source_references(chunks: List[Dict[str, Any]]) -> str:
    refs: List[str] = []
    for chunk in chunks:
        refs.append(f"{chunk.get('source_name', 'unknown')} p.{chunk.get('page_number', '?')}")
    return "; ".join(dict.fromkeys(refs))


def build_specialized_prompt(question: str, evidence: str) -> str:
    return (
        "You are a specialized scientific literature-analysis assistant. Use only the evidence provided to answer the question. "
        "Do not invent citations, page numbers, references, statistics, or findings. If the evidence is insufficient, say so clearly.\n\n"
        "Return a structured answer with these sections: Findings, Direct evidence, Limitations, and Unresolved questions.\n\n"
        f"Evidence:\n{evidence}\n\nResearch question:\n{question}"
    )


def run_specialized_pipeline(question: str, documents: List[Dict[str, Any]], config: Any, top_k: int = 5) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    for document in documents:
        chunks.extend(document.get("chunks", []))

    if not chunks:
        raise ValueError("No PDF text was extracted. Upload a readable PDF to enable the specialized pipeline.")

    retrieved = retrieve_relevant_chunks(question, chunks, config.embedding_model, top_k=top_k)
    evidence_text = "\n\n".join(
        f"{chunk.get('source_name', 'unknown')} (p.{chunk.get('page_number', '?')}): {chunk.get('text', '')}"
        for chunk in retrieved
    ) or "No relevant passages were retrieved from the uploaded document."

    system_prompt = (
        "You are a specialized scientific literature-analysis assistant. Use only the evidence provided, distinguish direct evidence from inference, "
        "and explicitly state uncertainty when the uploaded document does not contain enough support."
    )
    user_prompt = build_specialized_prompt(question, evidence_text)

    start = time.time()
    result = call_chat_completion(config, system_prompt, user_prompt, max_tokens=900)
    answer = result["answer"]
    return {
        "pipeline_name": "specialized",
        "answer": answer,
        "model_name": result["model_name"],
        "execution_time": round(time.time() - start + result["execution_time"], 2),
        "usage": result.get("usage", {}),
        "retrieved_chunks": retrieved,
        "retrieved_evidence": evidence_text,
        "source_references": format_source_references(retrieved),
        "top_k": top_k,
        "status": "success",
        "retrieval_metadata": {
            "chunks_available": len(chunks),
            "chunks_retrieved": len(retrieved),
            "embedding_model": config.embedding_model,
        },
    }
