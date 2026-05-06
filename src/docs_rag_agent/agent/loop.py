from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from docs_rag_agent.agent.tools import execute_search
from docs_rag_agent.llm import LLMClient, Message
from docs_rag_agent.retrieve import Reranker, SearchResult, VectorStore


@dataclass
class AgentStep:
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    final_answer: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    sources: list[SearchResult]
    model: str
    input_tokens: int
    output_tokens: int


MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a ReAct agent answering questions about FastAPI documentation.

You have access to ONE tool:

  search_docs(query: str, top_k: int = 5)
  Searches the FastAPI documentation and returns relevant text chunks.

On EVERY turn, respond with ONLY a JSON object — no prose, no markdown fences.

When you need to search, use:
{"thought": "<your reasoning>", "action": "search_docs",
 "action_input": {"query": "<query>", "top_k": 5}}

When you have enough information to answer, use:
{"thought": "<your reasoning>", "final_answer": "<complete answer citing source files>"}

Rules:
- Always call search_docs at least once before giving a final_answer.
- If an observation says all results were already seen, REFORMULATE substantially
  (different keywords, different angle) — do not repeat near-identical queries.
- If you cannot make progress after a couple of searches, give a partial
  final_answer with whatever you have rather than burning more iterations.
- Cite source files in your answer using [filename.md] notation.

Final-answer quality requirements:
- For "how do I ..." / "how to ..." questions, include a concrete code example
  copied (or closely adapted) from the observed chunks — not just a verbal
  description. Use a fenced ```python block.
- Structure the final_answer as:
    1. One- or two-sentence explanation of the approach.
    2. A code example in a ```python fenced block.
    3. A short "Source: [filename.md]" line (one or more files).
- Prefer completeness over brevity: if multiple steps are needed (e.g. define a
  dependency, then use it in a path operation), show all of them.
- Do not invent code that is not supported by the observations. If a needed
  detail is missing, say so explicitly instead of guessing."""


def _source_key(r: SearchResult) -> tuple[str, str, str]:
    return (r.metadata.get("source", ""), r.metadata.get("heading", ""), r.text[:64])


def _extract_json(text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if "```" in clean_text:
        # Find content between first pair of fences
        parts = clean_text.split("```")
        if len(parts) >= 3:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:].strip()
            clean_text = content.strip()
    return cast(dict[str, Any], json.loads(clean_text))


AgentEvent = tuple[Literal["step"], AgentStep] | tuple[Literal["final"], AgentResult]


def run_agent_iter(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    max_iterations: int = MAX_ITERATIONS,
    reranker: Reranker | None = None,
) -> Iterator[AgentEvent]:
    """Streaming version of run_agent: yields ("step", AgentStep) events as
    each ReAct iteration completes, then exactly one ("final", AgentResult).
    """
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"Question: {question}"),
    ]
    steps: list[AgentStep] = []
    all_sources: list[SearchResult] = []
    seen_keys: set[tuple[str, str, str]] = set()
    consecutive_empty = 0
    total_input = 0
    total_output = 0
    model_name = ""

    for _ in range(max_iterations):
        # 1024 matches the query-mode budget. Intermediate ReAct turns emit
        # a small JSON action and won't use most of it; final_answer turns
        # need the room to include a code example + citations.
        response = llm.generate(messages, max_tokens=1024, temperature=0.0)
        total_input += response.input_tokens
        total_output += response.output_tokens
        model_name = response.model

        try:
            raw = _extract_json(response.content)
            thought = str(raw.get("thought", ""))
        except json.JSONDecodeError as e:
            step = AgentStep(
                thought="System: Failed to parse previous JSON output.",
                observation=(
                    f"Invalid JSON produced: {e}. "
                    "Ensure you ONLY output valid JSON without any leading text."
                ),
            )
            steps.append(step)
            yield ("step", step)
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=f"Observation: {step.observation}"))
            continue

        if "final_answer" in raw:
            step = AgentStep(thought=thought, final_answer=str(raw["final_answer"]))
            steps.append(step)
            yield ("step", step)
            yield (
                "final",
                AgentResult(
                    answer=str(raw["final_answer"]),
                    steps=steps,
                    sources=all_sources,
                    model=model_name,
                    input_tokens=total_input,
                    output_tokens=total_output,
                ),
            )
            return
        else:
            action = str(raw.get("action", ""))
            action_input = raw.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            step = AgentStep(thought=thought, action=action, action_input=action_input)

            if action == "search_docs":
                query = str(action_input.get("query", question))
                top_k = int(action_input.get("top_k", 3))
                results = execute_search(store, query, top_k, reranker=reranker)

                new_results: list[SearchResult] = []
                for r in results:
                    key = _source_key(r)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        new_results.append(r)
                        all_sources.append(r)

                if not new_results:
                    consecutive_empty += 1
                    step.observation = (
                        f"All {len(results)} results from this query were already "
                        f"seen in previous searches. Reformulate with different "
                        f"keywords, or commit to a final_answer with what you have."
                    )
                else:
                    consecutive_empty = 0
                    obs_list = [
                        {
                            "source": r.metadata.get("source", ""),
                            "heading": r.metadata.get("heading", ""),
                            "text": r.text[:1200],
                        }
                        for r in new_results
                    ]
                    step.observation = json.dumps(obs_list)

                if consecutive_empty >= 2:
                    steps.append(step)
                    yield ("step", step)
                    yield (
                        "final",
                        AgentResult(
                            answer=(
                                "Stopped early: two consecutive searches returned no "
                                "new sources. Try a more specific question."
                            ),
                            steps=steps,
                            sources=all_sources,
                            model=model_name,
                            input_tokens=total_input,
                            output_tokens=total_output,
                        ),
                    )
                    return
            else:
                step.observation = f"Unknown tool: {action}"

            steps.append(step)
            yield ("step", step)
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=f"Observation: {step.observation}"))

    yield (
        "final",
        AgentResult(
            answer="Could not find a definitive answer within the iteration limit.",
            steps=steps,
            sources=all_sources,
            model=model_name,
            input_tokens=total_input,
            output_tokens=total_output,
        ),
    )


def run_agent(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    max_iterations: int = MAX_ITERATIONS,
    reranker: Reranker | None = None,
) -> AgentResult:
    final: AgentResult | None = None
    for kind, payload in run_agent_iter(
        question=question,
        store=store,
        llm=llm,
        max_iterations=max_iterations,
        reranker=reranker,
    ):
        if kind == "final":
            final = cast(AgentResult, payload)
    assert final is not None, "run_agent_iter must yield a final event"
    return final
