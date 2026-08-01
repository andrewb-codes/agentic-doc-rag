import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    internal_api_key: str
    telegram_bot_token: str
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


bot_settings = BotSettings()  # type: ignore[call-arg]
