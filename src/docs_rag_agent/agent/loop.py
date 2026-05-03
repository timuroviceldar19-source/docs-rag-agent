from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from docs_rag_agent.agent.tools import execute_search
from docs_rag_agent.llm import LLMClient, Message
from docs_rag_agent.retrieve import SearchResult, VectorStore


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

  search_docs(query: str, top_k: int = 3)
  Searches the FastAPI documentation and returns relevant text chunks.

On EVERY turn, respond with ONLY a JSON object — no prose, no markdown fences.

When you need to search, use:
{"thought": "<your reasoning>", "action": "search_docs", 
 "action_input": {"query": "<query>", "top_k": 3}}

When you have enough information to answer, use:
{"thought": "<your reasoning>", "final_answer": "<complete answer citing source files>"}

Always call search_docs at least once before giving a final_answer.
Cite source files in your answer using [filename.md] notation."""


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


def run_agent(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResult:
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"Question: {question}"),
    ]
    steps: list[AgentStep] = []
    all_sources: list[SearchResult] = []
    total_input = 0
    total_output = 0
    model_name = ""

    for _ in range(max_iterations):
        response = llm.generate(messages, max_tokens=512, temperature=0.0)
        total_input += response.input_tokens
        total_output += response.output_tokens
        model_name = response.model

        raw = _extract_json(response.content)
        thought = str(raw.get("thought", ""))

        if "final_answer" in raw:
            step = AgentStep(thought=thought, final_answer=str(raw["final_answer"]))
            steps.append(step)
            return AgentResult(
                answer=str(raw["final_answer"]),
                steps=steps,
                sources=all_sources,
                model=model_name,
                input_tokens=total_input,
                output_tokens=total_output,
            )
        else:
            action = str(raw.get("action", ""))
            action_input = raw.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}
            
            step = AgentStep(thought=thought, action=action, action_input=action_input)
            
            if action == "search_docs":
                query = str(action_input.get("query", question))
                top_k = int(action_input.get("top_k", 3))
                results = execute_search(store, query, top_k)
                all_sources.extend(results)
                
                obs_list = []
                for r in results:
                    obs_list.append({
                        "source": r.metadata.get("source", ""),
                        "heading": r.metadata.get("heading", ""),
                        "text": r.text[:300]
                    })
                step.observation = json.dumps(obs_list)
            else:
                step.observation = f"Unknown tool: {action}"
            
            steps.append(step)
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=f"Observation: {step.observation}"))

    return AgentResult(
        answer="Could not find a definitive answer within the iteration limit.",
        steps=steps,
        sources=all_sources,
        model=model_name,
        input_tokens=total_input,
        output_tokens=total_output,
    )
