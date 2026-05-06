#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from docs_rag_agent.config import Settings
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.eval import (
    EvalRecord,
    EvalResult,
    compute_faithfulness,
    compute_hit,
    compute_reciprocal_rank,
    summarize,
)
from docs_rag_agent.llm import build_llm_client
from docs_rag_agent.retrieve import CrossEncoderReranker, VectorStore

app = typer.Typer()


@app.command()
def run(
    dataset: Path = typer.Option(Path("data/eval_dataset.json"), help="Path to eval dataset."),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve per question."),
    judge: bool = typer.Option(False, help="Run LLM-as-judge faithfulness evaluation."),
    rerank: bool = typer.Option(False, help="Apply cross-encoder reranker after retrieval."),
    fetch_k: int = typer.Option(20, help="Over-fetch size before reranking."),
    output: Path = typer.Option(Path(""), help="Save JSON results to this path (optional)."),
) -> None:
    """Evaluate retrieval quality and optionally LLM answer faithfulness."""
    settings = Settings()
    embedder = FastEmbedLocalEmbedder(settings.embedding_model)
    client = QdrantClient(url=settings.qdrant_url)
    store = VectorStore(
        client=client,
        collection=settings.qdrant_collection,
        embedder=embedder,
    )
    llm = build_llm_client(settings) if judge else None
    reranker = CrossEncoderReranker(model_name=settings.rerank_model) if rerank else None

    raw = json.loads(dataset.read_text(encoding="utf-8"))
    records = [EvalRecord(**item) for item in raw]
    results: list[EvalResult] = []

    for rec in records:
        if reranker is None:
            search_results = store.search(rec.question, top_k=top_k)
        else:
            candidates = store.search(rec.question, top_k=max(fetch_k, top_k))
            search_results = reranker.rerank(rec.question, candidates, top_k=top_k)
        hit = compute_hit(search_results, rec.expected_source_contains)
        rr = compute_reciprocal_rank(search_results, rec.expected_source_contains)

        faith: float | None = None
        if judge and llm is not None and search_results:
            context = "\n\n".join(r.text[:300] for r in search_results)
            # Use a placeholder answer (reference) for offline eval
            faith = compute_faithfulness(
                question=rec.question,
                context=context,
                answer=rec.reference_answer,
                llm=llm,
            )

        results.append(
            EvalResult(
                question_id=rec.id,
                question=rec.question,
                hit=hit,
                reciprocal_rank=rr,
                faithfulness=faith,
            )
        )

        label = "HIT " if hit else "MISS"
        faith_str = f"  faithfulness={faith:.2f}" if faith is not None else ""
        typer.echo(f"[{label}] RR={rr:.2f}{faith_str}  {rec.question[:70]}", err=True)

    summary = summarize(results)
    typer.echo(f"\nHit Rate @ {top_k}:    {summary['hit_rate']:.2%}", err=True)
    typer.echo(f"MRR @ {top_k}:          {summary['mrr']:.3f}", err=True)
    if summary["mean_faithfulness"] > 0.0:
        typer.echo(f"Mean Faithfulness: {summary['mean_faithfulness']:.3f}", err=True)

    if output and str(output):
        payload = {
            "summary": summary,
            "top_k": top_k,
            "results": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "hit": r.hit,
                    "reciprocal_rank": r.reciprocal_rank,
                    "faithfulness": r.faithfulness,
                }
                for r in results
            ],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"Results saved to {output}", err=True)


if __name__ == "__main__":
    app()
