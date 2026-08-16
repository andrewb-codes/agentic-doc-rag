# Деплой через Ansible

Playbook деплоит приложение на один Ubuntu/Debian VPS. Сервер загружает готовые 
API/Bot images из GHCR и запускает пять сервисов через Docker Compose:

- `redis` — Redis для счетчиков rate limiting во внутренней сети;
- `postgres` — PostgreSQL 17 с volume `rag_postgres_data` во внутренней сети;
- `qdrant`  — Qdrant с volume `rag_qdrant_data` во внутренней сети
- `api` — FastAPI во внутренней сети;
- `bot` — Telegram bot клиент.

Bot обращается к API по адресу `http://api:8000`.
API, PostgreSQL, Redis и Qdrant снаружи не публикуются.

## Конфигурация

- `inventory.ini.example` — пример inventory;
- `group_vars/portfolio/main.yml` — несекретные переменные;
- `group_vars/portfolio/vault.yml.example` — шаблон секретов;
- `templates/env.j2` — production `.env`;
- `templates/docker-compose.prod.yml.j2` — production Compose;
- `playbook.yml` — установка Docker, очистка неиспользуемых Docker-объектов
  без удаления volumes, миграции и запуск сервисов.

Production images:

```yaml
api_image: ghcr.io/andrewb-codes/agentic-doc-rag-api
bot_image: ghcr.io/andrewb-codes/agentic-doc-rag-bot
app_image_tag: main

postgres_image: mirror.gcr.io/library/postgres:17
redis_image: mirror.gcr.io/library/redis:7.4-alpine
qdrant_image: qdrant/qdrant:v1.18.0
```

PostgreSQL, Redis и Qdrant используют зеркало Docker Hub official images, чтобы VPS не
упирался в anonymous pull rate limit Docker Hub. При необходимости эти значения
можно заменить на `postgres:17`, `redis:7.4-alpine` и `qdrant/qdrant:v1.18.0` 
или на образы из своего registry.

## Что делает playbook

1. Устанавливает Docker Engine и Compose plugin.
2. Очищает неиспользуемые Docker-объекты без удаления volumes, чтобы освободить место перед pull.
3. Создаёт каталог `/opt/apps/rag`.
4. Рендерит production `.env` и `docker-compose.prod.yml`.
5. Загружает images, запускает PostgreSQL/Redis/Qdrant и Alembic-миграции.
6. Поднимает API и Bot.
7. Повторно очищает неиспользуемые Docker-объекты после замены контейнеров.

Основные настройки находятся в `group_vars/portfolio/main.yml`, 
секреты — в зашифрованном `group_vars/portfolio/vault.yml`.

## Автоматический деплой

Workflow `.github/workflows/ci-cd.yml` запускает deploy после успешных проверок
при push в `main`. Images публикуются с тегами `main` и commit SHA.

GitHub Variables:

```text
VPS_HOST
VPS_USER
EMBEDDING_BASE_URL
LLM_BASE_URL
```

GitHub Secrets:

```text
VPS_SSH_KEY
INTERNAL_API_KEY
TELEGRAM_BOT_TOKEN
EMBEDDING_API_KEY
LLM_API_KEY
POSTGRES_PASSWORD
```

Публичная часть `VPS_SSH_KEY` должна находиться в `~/.ssh/authorized_keys` 
пользователя `VPS_USER`.

## Ручной запуск

```bash
cd deploy/ansible
cp inventory.ini.example inventory.ini
cp group_vars/portfolio/vault.yml.example group_vars/portfolio/vault.yml

# Заполнить inventory и vault.yml, затем:
ansible-vault encrypt group_vars/portfolio/vault.yml
ansible-playbook playbook.yml --ask-vault-pass
```

Для запуска без интерактивного ввода создайте файл с паролем Vault вне репозитория:

```bash
mkdir -p ~/.ansible
nano ~/.ansible/rag-vault-pass
chmod 600 ~/.ansible/rag-vault-pass
ansible-playbook playbook.yml \
  --vault-password-file ~/.ansible/rag-vault-pass
```

Ожидаемые поля Vault перечислены в `vault.yml.example`. 
Application images должны быть заранее опубликованы в GHCR.

## Диагностика

```bash
cd /opt/apps/rag
df -h
docker system df
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f bot
docker compose -f docker-compose.prod.yml logs -f postgres
docker compose -f docker-compose.prod.yml logs -f redis
docker compose -f docker-compose.prod.yml logs -f qdrant
docker compose -f docker-compose.prod.yml run --rm api alembic current
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
```

## Ротация пароля PostgreSQL

`POSTGRES_PASSWORD` применяется только при первичной инициализации пустого volume.
Для существующей БД сначала измените пароль роли:

```bash
cd /opt/apps/rag
docker compose -f docker-compose.prod.yml exec postgres \
  psql -U rag_user -d rag
```

В `psql`:

```text
\password rag_user
\q
```

Затем обновите `postgres_password` в Ansible Vault или `POSTGRES_PASSWORD` в GitHub
Secrets и повторите deploy. Удаление `rag_postgres_data` приводит к потере данных.

## Caddy

Telegram-бот работает через long polling и сам подключается к Telegram API.
Он не принимает входящие HTTP-запросы, поэтому публиковать порт и настраивать
Caddy для контейнера `bot` не требуется.
