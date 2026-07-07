"""model.py — Support RAG with guardrails and escalation for ChatAssist.

Pipeline: TF-IDF retrieval → extractive answer → guardrail check → escalation.
Structurally different from KnowledgeBot: adds guardrails + escalation logic.

Reference: Gao et al. (2023), "RAG for LLMs: A Survey".
"""
from __future__ import annotations
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been being with as at by "
    "this that these those it its from your you we our they their i do how what "
    "when where which why can should my me".split()
)
_PROFANITY = frozenset("stupid idiot dumb moron damn crap hell".split())
_THREATS = frozenset("hack steal breach attack destroy malware virus ransom".split())
_PII_REQUESTS = frozenset("phone number address salary ssn passport license".split())
_ESCALATION_KEYWORDS = frozenset("supervisor human agent manager legal complaint lawyer attorney sue".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_guardrails(text: str) -> dict[str, Any]:
    """Check input against profanity, threat, and PII request rules."""
    tokens = set(_tokenize(text))
    if tokens & _PROFANITY:
        return {"passed": False, "blocked_reason": "profanity_detected", "category": "profanity"}
    if tokens & _THREATS:
        return {"passed": False, "blocked_reason": "threat_detected", "category": "threat"}
    if tokens & _PII_REQUESTS:
        return {"passed": False, "blocked_reason": "pii_request", "category": "pii_request"}
    return {"passed": True, "blocked_reason": None, "category": None}


class SupportRetriever:
    """TF-IDF retriever with cosine similarity ranking over the FAQ corpus."""

    def __init__(self, ngram_range=(1, 2), min_df=1):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.doc_matrix = None
        self.documents: list[dict[str, Any]] = []

    def fit(self, documents: list[dict[str, Any]]) -> "SupportRetriever":
        self.documents = list(documents)
        contents = [f"{d.get('title', '')}. {d.get('content', '')}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df,
            sublinear_tf=True, norm="l2",
            tokenizer=_tokenize, token_pattern=None,
        )
        self.doc_matrix = self.vectorizer.fit_transform(contents)
        return self

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.doc_matrix.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["id"], "title": doc.get("title", ""),
                "content": doc.get("content", ""), "category": doc.get("category", ""),
                "score": float(scores[idx]),
            })
        return results

    def extractive_answer(self, query: str, k: int = 3) -> dict[str, Any]:
        retrieved = self.retrieve(query, k=k)
        if not retrieved:
            return {"answer": "", "source_doc_id": None, "context": "", "score": 0.0, "retrieved": []}
        top = retrieved[0]
        q_tokens = set(_tokenize(query))
        best_sent, best_overlap = "", 0
        for sent in _sentences(top["content"]):
            overlap = len(q_tokens & set(_tokenize(sent)))
            if overlap > best_overlap:
                best_sent, best_overlap = sent, overlap
        if not best_sent:
            best_sent = top["content"][:200]
        return {"answer": best_sent, "source_doc_id": top["doc_id"],
                "context": top["content"], "score": top["score"], "retrieved": retrieved}


# ── Full support pipeline: guardrails → retrieval → escalation ───────────────

def handle_query(retriever: SupportRetriever, query: str,
                 confidence_threshold: float = 0.05) -> dict[str, Any]:
    """Process a user query through guardrails, retrieval, and escalation logic.

    Returns dict with answer, source, escalated flag, guardrail status, and score.
    """
    guardrail = check_guardrails(query)
    if not guardrail["passed"]:
        return {
            "answer": "I cannot process this request. Let me connect you with a human agent.",
            "source_doc_id": None, "context": "", "score": 0.0,
            "escalated": True, "guardrail_passed": False,
            "blocked_reason": guardrail["blocked_reason"], "retrieved": [],
        }

    result = retriever.extractive_answer(query, k=3)
    query_tokens = set(_tokenize(query))
    wants_escalation = bool(query_tokens & _ESCALATION_KEYWORDS)

    if wants_escalation or result["score"] < confidence_threshold:
        return {
            "answer": "I am escalating your request to a human agent who will assist you shortly.",
            "source_doc_id": result.get("source_doc_id"),
            "context": result.get("context", ""), "score": result["score"],
            "escalated": True, "guardrail_passed": True,
            "blocked_reason": None, "retrieved": result.get("retrieved", []),
        }

    return {
        "answer": result["answer"], "source_doc_id": result["source_doc_id"],
        "context": result["context"], "score": result["score"],
        "escalated": False, "guardrail_passed": True,
        "blocked_reason": None, "retrieved": result.get("retrieved", []),
    }


def build_support_retriever(corpus: dict[str, Any]) -> SupportRetriever:
    """Convenience: fit a SupportRetriever on the corpus."""
    return SupportRetriever().fit(corpus["documents"])