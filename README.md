# agentic-doc-rag
Agentic Document RAG Assistant with Answer Verification

## План разработки

### Цель проекта

Сделать backend-сервис для вопросно-ответного поиска по техническим PDF-документам:

- загрузка PDF и извлечение текста
- разбиение текста на чанки с сохранением номеров страниц;
- vector search по фрагментам документа;
- генерация ответа через LLM только на основе найденного контекста;
- цитаты на использованные фрагменты;
- отдельный verifier-шаг, который проверяет, подтверждается ли ответ найденными чанками.

Главная идея проекта: сделать не просто PDF chatbot, а verified RAG-систему, которая умеет находить неподтвержденные или слабо подтвержденные ответы.

### Границы MVP

В первую версию нужно включить:

- FastAPI backend;
- парсинг PDF через PyMuPDF;
- recursive chunking с overlap;
- embeddings через `sentence-transformers`;
- хранение векторов в Qdrant;
- retrieval по конкретному документу через `document_id`;
- генерацию ответа через интерфейс `LLMProvider`;
- verifier agent с LLM-based и rule-based проверками;
- citations с metadata по страницам и чанкам;
- запуск через Docker Compose;
- pytest-тесты для ключевых сервисов.

PostgreSQL, Streamlit UI, hybrid search, reranking, Ollama support и RAG evaluation можно добавить после того, как заработает основной pipeline.

### Этап 1: Backend Skeleton

Собрать базовую структуру проекта:

- `app/main.py`;
- API routers;
- config module;
- schemas;
- services package;
- health endpoint;
- pytest setup;
- Dockerfile;
- `docker-compose.yml`.

Ожидаемый результат:

- `GET /health` возвращает статус сервиса;
- проект запускается локально;
- проект запускается через Docker Compose.

### Этап 2: PDF Ingestion

Реализовать загрузку документа и извлечение текста:

- `POST /documents/upload`;
- проверка, что загруженный файл является PDF;
- извлечение текста постранично через PyMuPDF;
- сохранение номеров страниц;
- создание metadata документа;
- возврат `document_id`, filename, status и page count.

Ожидаемый результат:

- пользователь может загрузить PDF;
- backend извлекает текст с page metadata;
- пустые или нечитаемые PDF обрабатываются корректно.

### Этап 3: Chunking

Реализовать recursive chunking:

- разбиение текста по разделам, абзацам, предложениям и лимиту размера;
- overlap между соседними чанками;
- сохранение `document_id`, `page`, `chunk_id`, `text` и `source`;
- тесты на размер чанков, overlap и сохранение metadata.

Ожидаемый результат:

- извлеченный текст PDF превращается в searchable chunks;
- каждый chunk можно связать с конкретной страницей документа.

### Этап 4: Embeddings и Qdrant

Добавить vector indexing:

- создать интерфейс `EmbeddingProvider`;
- реализовать `SentenceTransformerEmbedder`;
- позже добавить optional OpenAI-compatible embedder;
- создать Qdrant collection;
- сохранять chunk vectors вместе с payload metadata;
- фильтровать vectors по `document_id`.

Ожидаемый результат:

- чанки загруженного документа превращаются в embeddings;
- vectors сохраняются в Qdrant;
- payload содержит metadata для citations.

### Этап 5: Retrieval

Реализовать semantic retrieval:

- преобразовать вопрос пользователя в embedding;
- искать top-k чанков в Qdrant;
- фильтровать результаты по `document_id`;
- возвращать chunk text, page, chunk id, source и score.

Ожидаемый результат:

- `POST /documents/{document_id}/ask` может находить релевантные evidence chunks;
- retrieval отделен от answer generation и может тестироваться отдельно.

### Этап 6: Answer Agent

Реализовать генерацию ответа:

- создать интерфейс `LLMProvider`;
- реализовать первый OpenAI-compatible provider;
- составить prompt, который требует отвечать только по контексту;
- возвращать answer вместе с citations;
- явно обрабатывать ситуацию недостаточного контекста.

Ожидаемый результат:

- пользователь получает grounded answer;
- каждый ответ содержит citations на retrieved chunks;
- если информации в документе недостаточно, система отказывается отвечать вместо hallucination.

### Этап 7: Verifier Agent

Добавить verifier-шаг:

- передавать в verifier question, retrieved chunks, answer и citations;
- требовать structured JSON verdict;
- поддержать verdicts: `supported`, `partially_supported`, `unsupported`, `insufficient_context`;
- добавить rule-based checks для пустых citations, пустого retrieval и низких retrieval scores;
- возвращать unsupported claims и missing citations.

Ожидаемый результат:

- final response содержит verification verdict;
- неподтвержденные или слабо подтвержденные ответы помечаются явно;
- verifier становится главной отличительной фичей проекта.

### Этап 8: Завершение API

Довести основные endpoints:

- `POST /documents/upload`;
- `POST /documents/{document_id}/ask`;
- `GET /documents`;
- `GET /documents/{document_id}/chunks`;
- `GET /health`.

Ожидаемый результат:

- проект можно тестировать через Swagger UI;
- основные пользовательские сценарии работают без отдельного frontend.

### Этап 9: Тесты

Добавить focused test coverage:

- поведение chunking;
- edge cases при PDF extraction;
- фильтрация retrieval по документу;
- rule-based checks verifier-а;
- API health и validation tests.

Ожидаемый результат:

- core logic покрыта pytest-тестами;
- регрессии в chunking, retrieval и verification проще ловить.

### Этап 10: Документация и Demo

Улучшить презентацию проекта:

- architecture diagram;
- описание agentic workflow;
- примеры API endpoints;
- пример request и response;
- раздел про verification logic;
- инструкции запуска через Docker Compose;
- limitations и future improvements.

Ожидаемый результат:

- GitHub README понятно объясняет, почему это не обычный PDF chatbot;
- проект легко запустить и оценить.

### Future Improvements

Возможные улучшения после MVP:

- PostgreSQL для document metadata и QA logs;
- Streamlit demo UI;
- Ollama provider для локального LLM inference;
- OpenAI embeddings provider;
- hybrid search с BM25 и vector retrieval;
- reranker для улучшения выбора контекста;
- RAG evaluation через RAGAS, DeepEval или TruLens;
- OCR support для scanned PDFs;
- user accounts и document permissions;
- async background indexing для больших документов.

### Рекомендуемый порядок реализации

1. Backend skeleton.
2. PDF ingestion.
3. Chunking.
4. Embeddings и Qdrant indexing.
5. Retrieval.
6. Answer agent.
7. Verifier agent.
8. API cleanup.
9. Tests.
10. Documentation и demo examples.
