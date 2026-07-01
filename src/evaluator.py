from pathlib import Path
from typing import List, Dict, Any
import time

import pandas as pd

from retriever import query_knowledge_base, format_sources
from llm import generate_plain_answer, generate_answer
from graph import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "test_queries.csv"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.csv"


def split_keywords(keyword_text: str) -> List[str]:
    """
    Split expected keywords by comma.
    """
    if pd.isna(keyword_text):
        return []

    return [
        keyword.strip().lower()
        for keyword in str(keyword_text).split(",")
        if keyword.strip()
    ]


def score_keywords(answer: str, expected_keywords: str) -> float:
    """
    Calculate the proportion of expected keywords appearing in the answer.
    """
    keywords = split_keywords(expected_keywords)

    if not keywords:
        return 0.0

    lower_answer = answer.lower()

    matched = 0

    for keyword in keywords:
        if keyword.lower() in lower_answer:
            matched += 1

    return matched / len(keywords)


def score_source_hit(sources: List[str], expected_source: str) -> int:
    """
    Check whether expected source file appears in retrieved sources.
    """
    if pd.isna(expected_source) or not str(expected_source).strip():
        return 0

    expected_source = str(expected_source).lower()

    for source in sources:
        if expected_source in source.lower():
            return 1

    return 0


def run_plain_llm(query: str) -> Dict[str, Any]:
    """
    Version A: Plain LLM baseline.
    No retrieval.
    """
    start_time = time.time()

    answer = generate_plain_answer(query)

    latency = time.time() - start_time

    return {
        "answer": answer,
        "sources": [],
        "latency": latency,
        "retrieved_count": 0
    }


def run_pdf_ocr_rag(query: str) -> Dict[str, Any]:
    """
    Version B: PDF + OCR RAG baseline.
    This searches PDF text, PDF OCR, image OCR, and optional text records.
    """
    start_time = time.time()

    retrieved_docs = query_knowledge_base(
        query=query,
        n_results=5,
        allowed_modalities=["pdf_text_plus_ocr", "image_ocr", "text"]
    )

    answer = generate_answer(
        query=query,
        retrieved_docs=retrieved_docs,
        style_instruction=""
    )

    latency = time.time() - start_time

    return {
        "answer": answer,
        "sources": format_sources(retrieved_docs),
        "latency": latency,
        "retrieved_count": len(retrieved_docs)
    }


def run_final_agent(query: str) -> Dict[str, Any]:
    """
    Version C: Final PDF + OCR LangGraph Agent.
    """
    start_time = time.time()

    result = run_agent(query)

    latency = time.time() - start_time

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "latency": latency,
        "retrieved_count": len(result.get("retrieved_docs", []))
    }


def evaluate_system(
    system_name: str,
    query_id: str,
    family: str,
    query: str,
    expected_source: str,
    expected_keywords: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score one system output.
    """
    answer = result["answer"]
    sources = result["sources"]

    keyword_score = score_keywords(answer, expected_keywords)
    answer_success = 1 if keyword_score >= 0.5 else 0
    source_hit = score_source_hit(sources, expected_source)

    return {
        "query_id": query_id,
        "family": family,
        "system": system_name,
        "query": query,
        "expected_source": expected_source,
        "sources": " | ".join(sources),
        "source_hit": source_hit,
        "keyword_score": round(keyword_score, 3),
        "answer_success": answer_success,
        "latency_seconds": round(result["latency"], 3),
        "retrieved_count": result["retrieved_count"],
        "answer": answer
    }


def main() -> None:
    if not TEST_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find test queries file: {TEST_QUERIES_PATH}"
        )

    df = pd.read_csv(TEST_QUERIES_PATH)

    all_rows = []

    for _, row in df.iterrows():
        query_id = row["id"]
        family = row["family"]
        query = row["query"]
        expected_source = row["expected_source"]
        expected_keywords = row["expected_keywords"]

        print(f"\nEvaluating {query_id}: {query}")

        print("Running Plain LLM...")
        plain_result = run_plain_llm(query)
        all_rows.append(
            evaluate_system(
                "Plain LLM",
                query_id,
                family,
                query,
                expected_source,
                expected_keywords,
                plain_result
            )
        )

        print("Running PDF + OCR RAG...")
        pdf_ocr_rag_result = run_pdf_ocr_rag(query)
        all_rows.append(
            evaluate_system(
                "PDF + OCR RAG",
                query_id,
                family,
                query,
                expected_source,
                expected_keywords,
                pdf_ocr_rag_result
    )
)

        print("Running Final PDF + OCR Agent...")
        final_agent_result = run_final_agent(query)
        all_rows.append(
            evaluate_system(
                "Final PDF + OCR Agent",
                query_id,
                family,
                query,
                expected_source,
                expected_keywords,
                final_agent_result
            )
        )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    print("\nEvaluation completed.")
    print(f"Results saved to: {RESULTS_PATH}")

    summary = results_df.groupby("system").agg(
        avg_source_hit=("source_hit", "mean"),
        avg_keyword_score=("keyword_score", "mean"),
        avg_answer_success=("answer_success", "mean"),
        avg_latency_seconds=("latency_seconds", "mean")
    ).reset_index()

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()