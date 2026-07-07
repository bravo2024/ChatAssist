"""Smoke tests for ChatAssist — support RAG with guardrails and escalation."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_corpus
from src.model import SupportRetriever, build_support_retriever, handle_query, check_guardrails
from src.core import recall_at_k, mrr, containment_rate, escalation_rate, faithfulness


def test_corpus():
    """Support corpus has the right structure."""
    c = make_corpus()
    assert c["n_documents"] >= 20
    assert c["n_queries"] >= 15
    assert "guardrail_tests" in c
    assert "escalation_queries" in c


def test_retriever():
    """TF-IDF retriever returns ranked results."""
    c = make_corpus()
    r = build_support_retriever(c)
    results = r.retrieve("how do I reset my password", k=3)
    assert len(results) > 0
    assert results[0]["score"] >= results[-1]["score"]
    assert "CS-005" in [r["doc_id"] for r in results[:3]]  # password reset doc


def test_extractive_answer():
    """Extractive answer is grounded in the retrieved context."""
    c = make_corpus()
    r = build_support_retriever(c)
    ans = r.extractive_answer("how long does a refund take", k=3)
    assert ans["answer"] and len(ans["answer"]) > 10
    assert ans["answer"] in ans["context"]  # grounded


def test_guardrails():
    """Guardrails block profanity, threats, and PII requests."""
    assert not check_guardrails("you are a stupid bot")["passed"]
    assert not check_guardrails("I will hack your system")["passed"]
    assert not check_guardrails("what is the CEO phone number")["passed"]
    assert check_guardrails("how do I reset my password")["passed"]


def test_escalation():
    """Queries with escalation keywords are escalated."""
    c = make_corpus()
    r = build_support_retriever(c)
    result = handle_query(r, "I need to speak to a human supervisor")
    assert result["escalated"] is True


def test_guardrail_block_escalates():
    """Guardrail-blocked queries are escalated."""
    c = make_corpus()
    r = build_support_retriever(c)
    result = handle_query(r, "you are a stupid bot")
    assert result["escalated"] is True
    assert result["guardrail_passed"] is False


def test_normal_query_not_escalated():
    """Normal queries are answered without escalation."""
    c = make_corpus()
    r = build_support_retriever(c)
    result = handle_query(r, "how do I reset my password")
    assert result["escalated"] is False
    assert result["guardrail_passed"] is True
    assert len(result["answer"]) > 0


def test_retrieval_metrics():
    """Recall@k and MRR compute correctly on the eval queries."""
    c = make_corpus()
    r = build_support_retriever(c)
    recalls = []
    for q in c["queries"][:10]:
        retrieved = [x["doc_id"] for x in r.retrieve(q["query"], k=5)]
        recalls.append(recall_at_k(retrieved, q["relevant_doc_ids"], 5))
    assert sum(recalls) / len(recalls) > 0.3  # TF-IDF should retrieve well


def test_containment_and_escalation_rates():
    """Containment and escalation rates compute from handle_query results."""
    c = make_corpus()
    r = build_support_retriever(c)
    results = [handle_query(r, q["query"]) for q in c["queries"]]
    cont = containment_rate(results)
    esc = escalation_rate(results)
    assert 0.0 <= cont <= 1.0
    assert 0.0 <= esc <= 1.0
    assert abs(cont + esc - 1.0) < 1e-9  # every query is either contained or escalated


if __name__ == "__main__":
    test_corpus()
    test_retriever()
    test_extractive_answer()
    test_guardrails()
    test_escalation()
    test_guardrail_block_escalates()
    test_normal_query_not_escalated()
    test_retrieval_metrics()
    test_containment_and_escalation_rates()
    print("All ChatAssist smoke tests passed!")
