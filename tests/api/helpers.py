from agentic_rag.core.config import settings


def internal_headers(
    *,
    telegram_user_id: int = 123456789,
    telegram_username: str | None = "andrew",
    internal_api_key: str = settings.internal_api_key,
) -> dict[str, str]:
    headers = {
        "X-Internal-API-Key": internal_api_key,
        "X-Telegram-User-Id": str(telegram_user_id),
    }

    if telegram_username is not None:
        headers["X-Telegram-Username"] = telegram_username

    return headers
