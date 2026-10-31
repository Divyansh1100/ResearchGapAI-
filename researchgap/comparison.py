from __future__ import annotations

from typing import Dict, List

EVALUATION_DIMENSIONS = [
    "accuracy",
    "relevance",
    "evidence_traceability",
    "completeness",
    "research_gap_quality",
    "consistency",
]


def score_summary(scores: Dict[str, float]) -> Dict[str, float]:
    numeric_scores = []
    for dimension in EVALUATION_DIMENSIONS:
        value = scores.get(dimension)
        if isinstance(value, (int, float)):
            numeric_scores.append(float(value))
    total = sum(numeric_scores)
    average = total / len(numeric_scores) if numeric_scores else 0.0
    return {"total": round(total, 2), "average": round(average, 2)}


def empty_score_state() -> Dict[str, Dict[str, int]]:
    return {
        "general": {dimension: 0 for dimension in EVALUATION_DIMENSIONS},
        "specialized": {dimension: 0 for dimension in EVALUATION_DIMENSIONS},
    }


def evaluation_dataframe(scores: Dict[str, Dict[str, int]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for pipeline_name in ["general", "specialized"]:
        pipeline_scores = scores.get(pipeline_name, {})
        summary = score_summary(pipeline_scores)
        rows.append({
            "pipeline": pipeline_name,
            "total": summary["total"],
            "average": summary["average"],
            **pipeline_scores,
        })
    return rows
