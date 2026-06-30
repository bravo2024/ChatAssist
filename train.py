"""train.py — Build the support RAG index and evaluate retrieval + escalation."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import make_corpus
from src.model import build_support_retriever
from src.evaluate import evaluate_support, save_metrics, print_report


def main():
    corpus = make_corpus()
    print(f"Corpus: {corpus['n_documents']} docs, {corpus['n_queries']} queries")
    print(f"Categories: {', '.join(corpus['categories'])}")

    retriever = build_support_retriever(corpus)
    print("Built TF-IDF support retriever.")

    metrics = evaluate_support(retriever, corpus["queries"])
    print_report(metrics)
    save_metrics(metrics)
    print("\nSaved metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
