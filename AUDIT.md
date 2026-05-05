# Аудит и рефакторинг проекта docs-rag-agent

## Список выявленных проблем и статус

- [x] **[blocking]** Ошибка парсинга JSON в `src/docs_rag_agent/agent/loop.py` вызывает 500 ошибку. Необходимо добавить обработку `json.JSONDecodeError` и fallback-механизм (возврат ошибки агенту для исправления).
- [x] **[blocking]** Отсутствие аутентификации на эндпоинтах `POST /query` и `POST /agent` в `src/docs_rag_agent/api/main.py`. Необходим `APIKeyHeader`.
- [x] **[serious]** Утечка потенциально чувствительной информации в `_handle_llm_error` (`src/docs_rag_agent/api/main.py`), отдается сырой текст `exc`. Необходимо заменить на статичное сообщение.
- [x] **[serious]** Отсутствие батчинга в `VectorStore.upsert` (`src/docs_rag_agent/retrieve/store.py`), что может вызвать OOM или таймаут. Необходимо добавить `itertools.batched`.

## Лог работы
*Файл создан.*
*Добавлена поддержка LLM провайдера Groq в `src/docs_rag_agent/llm/factory.py` и `src/docs_rag_agent/config.py`.*
