"""core.py — Retrieval, guardrail, and escalation metrics for ChatAssist.

These are support-automation metrics, NOT generic classification metrics:
  * **containment_rate** — fraction of queries answered without escalation.
  * **escalation_rate** — fraction routed to a human agent.
  * **answer_hit_rate** — fraction of queries that retrieved at least one
    relevant passage above a confidence threshold.
  * **guardrail_pass_rate** — fraction of responses that pass safety checks.
  * **faithfulness** — SQuAD-style token-F1 between answer and retrieved context.

Reference
---------
Gao et al. (2023), "Retrieval-Augmented Generation for Large Language Models:
A Survey"; also Faggella (2023) on containment/deflection KPIs.
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been being with as at by "
    "this that these those it its from your you we our they their i do how what "
    "when where which why can should my me".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


# ── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    top = list(retrieved_ids)[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    top = list(retrieved_ids)[:k]
    return sum(1 for r in top if r in relevant) / k


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


# ── Faithfulness (SQuAD-style token F1) ──────────────────────────────────────

def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, context: str) -> float:
    """Token-F1 overlap between the answer and the retrieved context passage."""
    return _f1(_tokenize(answer), _tokenize(context))


def max_faithfulness(answer: str, contexts: list[str]) -> float:
    """Best token-F1 between the answer and any retrieved passage."""
    a_tokens = _tokenize(answer)
    if not a_tokens:
        return 0.0
    return max((_f1(a_tokens, _tokenize(c)) for c in contexts), default=0.0)


# ── Support automation KPIs ──────────────────────────────────────────────────

def containment_rate(results: list[dict]) -> float:
    """Fraction of queries answered without escalation (higher = better deflection)."""
    if not results:
        return 0.0
    return sum(1 for r in results if not r.get("escalated", False)) / len(results)


def escalation_rate(results: list[dict]) -> float:
    """Fraction of queries routed to a human agent."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("escalated", False)) / len(results)


def answer_hit_rate(results: list[dict], confidence_threshold: float = 0.15) -> float:
    """Fraction of queries with at least one passage above confidence threshold."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("top_score", 0) >= confidence_threshold) / len(results)


def guardrail_pass_rate(results: list[dict]) -> float:
    """Fraction of responses that passed all guardrail checks."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("guardrail_passed", True)) / len(results)