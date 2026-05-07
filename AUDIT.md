# Аудит и рефакторинг проекта docs-rag-agent

## Список выявленных проблем и статус

- [x] **[blocking]** Ошибка парсинга JSON в `src/docs_rag_agent/agent/loop.py` вызывает 500 ошибку. Необходимо добавить обработку `json.JSONDecodeError` и fallback-механизм (возврат ошибки агенту для исправления).
- [x] **[blocking]** Отсутствие аутентификации на эндпоинтах `POST /query` и `POST /agent` в `src/docs_rag_agent/api/main.py`. Необходим `APIKeyHeader`.
- [x] **[serious]** Утечка потенциально чувствительной информации в `_handle_llm_error` (`src/docs_rag_agent/api/main.py`), отдается сырой текст `exc`. Необходимо заменить на статичное сообщение.
- [x] **[serious]** Отсутствие батчинга в `VectorStore.upsert` (`src/docs_rag_agent/retrieve/store.py`), что может вызвать OOM или таймаут. Необходимо добавить `itertools.batched`.

## Лог работы
*Файл создан.*
*Добавлена поддержка LLM провайдера Groq в `src/docs_rag_agent/llm/factory.py` и `src/docs_rag_agent/config.py`.*
*Подкручен ReAct-промпт и token budget агента (512 → 1024 на ход, обсервации до 1200 символов вместо 300, требование code-example в final_answer). Без реранкера на 10-item golden set: Hit Rate@5 80%, MRR@5 0.46, Mean Faithfulness 0.93.*
*Добавлен cross-encoder reranker (`BAAI/bge-reranker-base`) как второй stage retrieval'а — в `src/docs_rag_agent/retrieve/reranker.py`, проброшен через `/query`, `/agent` и `scripts/eval.py --rerank`. Конфиг через `RERANK_ENABLED`/`RERANK_MODEL`/`RERANK_FETCH_K`. С реранкером: Hit Rate@5 100%, MRR@5 0.56. Eval JSON для baseline и with-reranker под `eval_results/`.*
*SSE стриминг для `/query` и `/agent`. В `LLMClient` Protocol добавлен `generate_stream`, реализации для OpenAI/Groq, Gemini, Anthropic. Агентский loop отрефакторен в генератор `run_agent_iter` (старый `run_agent` стал тонким редьюсером поверх него — без дубля логики). Новые эндпоинты `/query/stream` (chunks → token… → end) и `/agent/stream` (step… → final). Ошибки приходят как `event: error` внутри стрима, статус 200. Streamlit получил toggle "Stream response". 9 новых тестов в `tests/test_streaming.py`.*
*Hybrid retrieval (dense + BM25 + RRF). Добавлен `FastEmbedSparseEmbedder` (`Qdrant/bm25`), `VectorStore` переехал на named-vector layout (`dense` cosine + sparse `bm25` с `Modifier.IDF`), `search` использует server-side RRF через `Prefetch + FusionQuery`. Конфиг: `HYBRID_ENABLED`/`SPARSE_MODEL`/`HYBRID_FETCH_K`. На том же 10-item golden set: hybrid alone 70%/0.54, hybrid + rerank 80%/0.517 — на этой natural-language выборке dense + rerank по-прежнему лучший (100%/0.557). Hybrid оставлен как production-ready путь для keyword-heavy запросов; цифры все четыре в README, выбор честный. 6 новых тестов в `tests/test_hybrid_retrieval.py`. Старая dense-only коллекция несовместима — нужен drop + re-ingest.*
