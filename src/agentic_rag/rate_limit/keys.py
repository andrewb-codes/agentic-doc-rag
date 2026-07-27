from agentic_rag.rate_limit.scopes import RateLimitScope

KEY_NAMESPACE = "rate-limit"


def build_user_key(*, scope: RateLimitScope, user_id: int) -> str:
    return f"{KEY_NAMESPACE}:{scope}:user:{user_id}"
