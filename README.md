# Agentic Doc RAG

Backend и Telegram-бот для вопросно-ответного поиска по PDF-документам. Проект
реализует verified RAG pipeline: документ индексируется в Qdrant, ответ генерируется
только по найденному контексту, а отдельный verifier-шаг проверяет, подтверждается ли
ответ фрагментами документа.

## Возможности

- загрузка PDF через Telegram;
- извлечение текста по страницам через PyMuPDF;
- chunking с overlap и сохранением page/source metadata;
- хранение пользователей, документов, чанков и истории вопросов в PostgreSQL;
- OpenAI-compatible embeddings и LLM provider;
- semantic search по Qdrant с фильтрацией по владельцу и документу;
- ответы по одному активному документу или по всем документам пользователя;
- verification verdict для каждого ответа: `supported` или `unsupported`;
- история вопросов и ответов;
- Telegram UI для загрузки, выбора и удаления документов;
- rate limiting через Redis;
- структурированное логирование запросов с `X-Request-ID`;
- Docker Compose для API, bot, PostgreSQL, Redis и Qdrant;
- pytest-покрытие сервисов, API, repositories, rate limiting и Telegram client.

Основной стек: Python 3.13, FastAPI, aiogram 3, SQLAlchemy 2 async, PostgreSQL,
Qdrant, Redis, Alembic, PyMuPDF, OpenAI SDK, structlog, Docker Compose и uv.

## Архитектура

FastAPI содержит бизнес-логику и RAG pipeline. Telegram-бот является отдельным
HTTP-клиентом backend API и не обращается напрямую к PostgreSQL или Qdrant.

```text
Telegram user
    │
    ▼
Telegram bot ── internal HTTP ──▶ FastAPI
                                  │
                                  ├── SQLAlchemy ──▶ PostgreSQL
                                  ├── embeddings/search ──▶ Qdrant
                                  ├── rate limits ──▶ Redis
                                  └── OpenAI-compatible LLM provider
```

RAG-сценарий:

```text
PDF upload
  -> text extraction
  -> chunking
  -> embeddings
  -> Qdrant indexing

question
  -> ownership check
  -> semantic retrieval
  -> answer generation
  -> answer verification
  -> QA history
```

В Docker Compose API, bot, PostgreSQL, Redis и Qdrant находятся во внутренней
сети. API не публикуется на хост; пользовательский интерфейс MVP — Telegram-бот.

## Локальный запуск через Docker Compose

Создайте `.env`:

```bash
cp .env.example .env
```

Замените placeholder-значения:

```env
DATABASE_URL=postgresql+asyncpg://rag_user:replace-with-db-password@postgres:5432/rag
INTERNAL_API_KEY=replace-with-a-long-random-internal-secret
TELEGRAM_BOT_TOKEN=replace-with-your-telegram-bot-token
EMBEDDING_API_KEY=replace-with-your-api-key
LLM_API_KEY=replace-with-your-api-key
POSTGRES_DB=rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=replace-with-db-password
POSTGRES_PORT=5432
```

В `.env` значение `DATABASE_URL` рассчитано на Docker Compose, поэтому host базы
данных — `postgres`. Для локального запуска тестов используется отдельный
`.env.test`, где database URL указывает на `localhost`.

Если используется не OpenAI напрямую, добавьте OpenAI-compatible endpoints и модель:

```env
EMBEDDING_BASE_URL=https://your-embedding-provider/v1
LLM_BASE_URL=https://your-llm-provider/v1
LLM_MODEL=gpt-4o-mini
```

Для медленных LLM-провайдеров полезно увеличить timeout:

```env
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=1
```

Первый запуск:

```bash
docker compose up --build -d postgres redis qdrant
docker compose run --rm api alembic upgrade head
docker compose up -d api telegram-bot
docker compose logs -f api
```

При следующих запусках, если миграции не менялись:

```bash
docker compose up --build -d
```

## API

API рассчитан на доверенного клиента внутри инфраструктуры. Запросы требуют:

```text
X-Internal-Api-Key: <INTERNAL_API_KEY>
X-Telegram-User-Id: <telegram user id>
X-Telegram-Username: <optional username>
```

| Method | Route | Назначение |
|---|---|---|
| `GET` | `/health` | Healthcheck |
| `GET` | `/documents` | Документы текущего Telegram-пользователя |
| `POST` | `/documents/upload` | Загрузка и индексация PDF |
| `GET` | `/documents/{document_id}` | Метаданные документа |
| `DELETE` | `/documents/{document_id}` | Удаление документа |
| `POST` | `/documents/search` | Поиск чанков по всем документам пользователя |
| `POST` | `/documents/{document_id}/search` | Поиск чанков внутри документа |
| `POST` | `/documents/ask` | Ответ по всем документам пользователя |
| `POST` | `/documents/{document_id}/ask` | Ответ по выбранному документу |
| `GET` | `/qa-history` | История вопросов пользователя |

Пример ответа `/documents/{document_id}/ask`:

```json
{
  "answer": "Документ описывает FastAPI backend и Qdrant vector search.",
  "answer_status": "answered",
  "chunks": [
    {
      "id": 17,
      "document_id": 3,
      "page": 4,
      "chunk_index": 12,
      "text": "...",
      "source": "manual.pdf"
    }
  ],
  "verification_result": {
    "verdict": "supported",
    "unsupported_claims": [],
    "missing_information": [],
    "confidence": 0.92
  }
}
```

Если verifier находит неподтвержденное утверждение, элемент
`unsupported_claims` содержит сам claim и короткую причину:

```json
"unsupported_claims": [
  {
    "claim": "Документ использует Redis как основное хранилище данных.",
    "reason": "В найденных фрагментах Redis описан только как storage для rate limiting."
  }
]
```

Ошибки приложения возвращаются в едином формате:

```json
{"detail": "error.<domain>.<reason>"}
```

## Rate limiting

Rate limiting реализован через `limits` и Redis storage. В Compose Redis работает
во внутренней сети по адресу `async+redis://redis:6379/0`, который задается в
`docker-compose.yml`.

При превышении лимита API возвращает:

```json
{"detail": "error.rate_limit.exceeded"}
```

со статусом `429 Too Many Requests` и заголовком `Retry-After`. Для успешных
limited responses добавляются `X-RateLimit-Limit`, `X-RateLimit-Remaining` и
`X-RateLimit-Reset`.

Начальные лимиты:

| Scope | Route | Limit |
|---|---|---|
| `document_upload` | `POST /documents/upload` | `10 per hour` |
| `document_search` | `POST /documents/search`, `POST /documents/{document_id}/search` | `120 per hour` |
| `document_ask` | `POST /documents/ask`, `POST /documents/{document_id}/ask` | `60 per hour` |
| `qa_history_read` | `GET /qa-history` | `120 per hour` |

## Разработка

Установка всех runtime и dev-зависимостей:

```bash
uv sync --all-extras --all-groups
```

Установка только backend-зависимостей:

```bash
uv sync --extra api
```

Установка только Telegram bot-зависимостей:

```bash
uv sync --extra bot
```

Основные директории:

```text
src/agentic_rag/api/          FastAPI entrypoint, routers, presenters и dependencies
src/agentic_rag/bot/          aiogram bot, backend client и форматирование сообщений
src/agentic_rag/core/         settings, logging, shared errors и enum-ы
src/agentic_rag/db/           SQLAlchemy base/session
src/agentic_rag/middleware/   request logging middleware
src/agentic_rag/models/       SQLAlchemy-модели
src/agentic_rag/repositories/ запросы к PostgreSQL
src/agentic_rag/services/     PDF, chunking, indexing, retrieval, answer и verification
src/agentic_rag/vectorstores/ Qdrant adapter
alembic/                      миграции
tests/                        unit, API, component и integration tests
```

Создание и применение миграции:

```bash
uv run alembic revision --autogenerate -m "describe schema change"
uv run alembic upgrade head
```

Применение миграции в Docker Compose:

```bash
docker compose run --rm api alembic upgrade head
```

## Тесты и проверки

Тесты используют отдельную PostgreSQL-базу `rag_test` в том же
PostgreSQL-контейнере. Не указывайте в `.env.test` основную БД `rag`:
фикстуры тестов очищают таблицы перед каждым тестом. Пользователь, пароль и
внешний порт в `DATABASE_URL` из `.env.test` должны соответствовать
`POSTGRES_USER`, `POSTGRES_PASSWORD` и `POSTGRES_PORT` в `.env`.

В `.env.test` rate limiting выключен через `RATE_LIMIT_ENABLED=false`, чтобы
обычный test suite не зависел от Redis. Rate-limit wiring тестируется отдельно
через FastAPI dependency overrides и fake service.

```bash
cp .env.test.example .env.test
docker compose up -d postgres
docker compose exec postgres psql -U rag_user -d rag \
  -c "CREATE DATABASE rag_test;"
ENV_FILE=.env.test uv run alembic upgrade head
ENV_FILE=.env.test uv run pytest
```

Статические проверки:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
```

## Ограничения MVP

- OCR для сканированных PDF не реализован;
- indexing выполняется синхронно во время upload;
- hybrid search, BM25 и reranking не добавлены;
- API не имеет публичной пользовательской аутентификации: текущий клиент — Telegram bot;

## Возможные улучшения

- background indexing через worker queue;
- OCR для scanned PDFs;
- retrieval score threshold и reranking;
- hybrid search;
- web frontend;
- RAG evaluation через RAGAS, DeepEval или TruLens;
- observability через OpenTelemetry или Langfuse.
