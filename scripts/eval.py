#!/usr/bin/env python
"""Eval CLI for docs-rag-agent.

Two modes:
- ``--mode query`` (default): pure retrieval eval. For each question runs
  Qdrant search (optionally + reranker) and computes Hit Rate / MRR. With
  ``--judge`` also asks an LLM to score how faithful the dataset's
  ``reference_answer`` is to the retrieved context.

- ``--mode agent``: end-to-end agent eval. For each question actually runs
  the ReAct loop and judges the *generated* answer against the *agent's
  retrieved sources*. Hit / MRR are computed against everything the agent
  pulled across its iterations. Slower and costs LLM tokens (multiple
  generate calls per question).
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from docs_rag_agent.agent.loop import run_agent
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
    mode: str = typer.Option(
        "query",
        help="`query` — retrieval-only eval. `agent` — run the ReAct loop end-to-end.",
    ),
    max_iterations: int = typer.Option(5, help="Max ReAct iterations (mode=agent only)."),
    output: Path = typer.Option(Path(""), help="Save JSON results to this path (optional)."),
) -> None:
    """Evaluate retrieval and (optionally) generated-answer quality."""
    if mode not in {"query", "agent"}:
        raise typer.BadParameter("--mode must be 'query' or 'agent'")

    settings = Settings()
    embedder = FastEmbedLocalEmbedder(settings.embedding_model)
    client = QdrantClient(url=settings.qdrant_url)
    store = VectorStore(
        client=client,
        collection=settings.qdrant_collection,
        embedder=embedder,
    )
    # Agent mode always needs the LLM. Query mode only needs it for --judge.
    llm = build_llm_client(settings) if (judge or mode == "agent") else None
    reranker = CrossEncoderReranker(model_name=settings.rerank_model) if rerank else None

    raw = json.loads(dataset.read_text(encoding="utf-8"))
    records = [EvalRecord(**item) for item in raw]
    results: list[EvalResult] = []

    for rec in records:
        if mode == "query":
            if reranker is None:
                search_results = store.search(rec.question, top_k=top_k)
            else:
                candidates = store.search(rec.question, top_k=max(fetch_k, top_k))
                search_results = reranker.rerank(rec.question, candidates, top_k=top_k)
            generated_answer: str | None = None
        else:
            assert llm is not None
            agent_result = run_agent(
                question=rec.question,
                store=store,
                llm=llm,
                max_iterations=max_iterations,
                reranker=reranker,
            )
            search_results = agent_result.sources
            generated_answer = agent_result.answer

        hit = compute_hit(search_results, rec.expected_source_contains)
        rr = compute_reciprocal_rank(search_results, rec.expected_source_contains)

        faith: float | None = None
        if judge and llm is not None and search_results:
            context = "\n\n".join(r.text[:300] for r in search_results)
            answer_to_judge = (
                generated_answer if generated_answer is not None else rec.reference_answer
            )
            faith = compute_faithfulness(
                question=rec.question,
                context=context,
                answer=answer_to_judge,
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
    typer.echo(f"\nMode: {mode}{' + rerank' if rerank else ''}", err=True)
    typer.echo(f"Hit Rate @ {top_k}:    {summary['hit_rate']:.2%}", err=True)
    typer.echo(f"MRR @ {top_k}:          {summary['mrr']:.3f}", err=True)
    if summary["mean_faithfulness"] > 0.0:
        suffix = " (generated answer)" if mode == "agent" else " (reference answer)"
        typer.echo(f"Mean Faithfulness: {summary['mean_faithfulness']:.3f}{suffix}", err=True)

    if output and str(output):
        payload = {
            "summary": summary,
            "mode": mode,
            "rerank": rerank,
            "top_k": top_k,
            "max_iterations": max_iterations if mode == "agent" else None,
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
