"""evaluate.py — Batch evaluation of the ChatAssist support RAG pipeline."""
from __future__ import annotations
import json
from pathlib import Path

from src.core import (
    recall_at_k, precision_at_k, mrr, containment_rate,
    escalation_rate, answer_hit_rate, guardrail_pass_rate, max_faithfulness,
)


def evaluate_support(retriever, queries, k_values=(1, 3, 5)):
    """Run all eval queries through the full pipeline and aggregate metrics."""
    from src.model import handle_query
    per_query = []
    for q in queries:
        result = handle_query(retriever, q["query"])
        retrieved_ids = [r["doc_id"] for r in result.get("retrieved", [])]
        retrieved_contexts = [r["content"] for r in result.get("retrieved", [])]
        per_query.append({
            "query_id": q["id"],
            "query": q["query"],
            "answer": result["answer"],
            "source_doc_id": result["source_doc_id"],
            "escalated": result["escalated"],
            "guardrail_passed": result["guardrail_passed"],
            "top_score": result["score"],
            "faithfulness": max_faithfulness(result["answer"], retrieved_contexts) if retrieved_contexts else 0.0,
            "recall@5": recall_at_k(retrieved_ids, q.get("relevant_doc_ids", []), 5),
            "mrr": mrr(retrieved_ids, q.get("relevant_doc_ids", [])),
        })
    aggregate = {
        "n_queries": len(per_query),
        "containment_rate": containment_rate(per_query),
        "escalation_rate": escalation_rate(per_query),
        "answer_hit_rate": answer_hit_rate(per_query),
        "guardrail_pass_rate": guardrail_pass_rate(per_query),
        "recall@5": sum(p["recall@5"] for p in per_query) / max(len(per_query), 1),
        "mrr": sum(p["mrr"] for p in per_query) / max(len(per_query), 1),
        "faithfulness": sum(p["faithfulness"] for p in per_query) / max(len(per_query), 1),
    }
    return {"aggregate": aggregate, "per_query": per_query}


def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _clean(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            return {k: _clean(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v
    with open(path, "w") as f:
        json.dump(_clean(metrics), f, indent=2)
    return metrics


def print_report(metrics):
    agg = metrics["aggregate"]
    print("=" * 50)
    print("  ChatAssist Support RAG Evaluation")
    print("=" * 50)
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.4f}")
        else:
            print(f"    {k:25s}: {v}")
