INTERNAL_API_KEY = "test-internal-api-key"


def internal_headers(
    *,
    telegram_user_id: int = 123456789,
    telegram_username: str | None = "andrew",
    internal_api_key: str = INTERNAL_API_KEY,
) -> dict[str, str]:
    headers = {
        "X-Internal-API-Key": internal_api_key,
        "X-Telegram-User-Id": str(telegram_user_id),
    }

    if telegram_username is not None:
        headers["X-Telegram-Username"] = telegram_username

    return headers
