from __future__ import annotations

import os
import time
from dataclasses import dataclass

import redis


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_s: int
    limit: int
    window_s: int


def _redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def _limits_for_route(route_key: str) -> tuple[int, int]:
    """Devuelve (limit, window_s) por route.

    Configurable por env:
      RATE_LIMIT_CHAT_PER_USER=30
      RATE_LIMIT_CHAT_WINDOW_S=60
    """
    if route_key.startswith("chat"):
        limit = int(os.getenv("RATE_LIMIT_CHAT_PER_USER", "30"))
        window_s = int(os.getenv("RATE_LIMIT_CHAT_WINDOW_S", "60"))
        return limit, window_s

    limit = int(os.getenv("RATE_LIMIT_DEFAULT", "120"))
    window_s = int(os.getenv("RATE_LIMIT_DEFAULT_WINDOW_S", "60"))
    return limit, window_s


def check_rate_limit(*, tenant_id: str, user_id: str, route_key: str) -> RateLimitDecision:
    """Rate limiting por ventana fija (tenant_id+user_id+route).

    Operabilidad:
    - Si Redis no está disponible y RATE_LIMIT_FAIL_OPEN=true (default), permite la request.
    - Si RATE_LIMIT_FAIL_OPEN=false, falla en 503.
    """
    limit, window_s = _limits_for_route(route_key)

    now = int(time.time())
    bucket = now // window_s
    key = f"rl:{os.getenv('APP_ENV', os.getenv('ENV','dev'))}:{route_key}:{tenant_id}:{user_id}:{bucket}"

    fail_open = os.getenv("RATE_LIMIT_FAIL_OPEN", "true").lower() in {"1", "true", "yes"}

    try:
        r = _redis_client()
        current = int(r.incr(key))
        if current == 1:
            r.expire(key, window_s + 5)
    except Exception:
        if fail_open:
            return RateLimitDecision(
                allowed=True,
                remaining=limit,
                retry_after_s=0,
                limit=limit,
                window_s=window_s,
            )
        return RateLimitDecision(
            allowed=False,
            remaining=0,
            retry_after_s=1,
            limit=limit,
            window_s=window_s,
        )

    remaining = max(0, limit - current)
    allowed = current <= limit

    window_end = (bucket + 1) * window_s
    retry_after = max(0, window_end - now)

    return RateLimitDecision(
        allowed=allowed,
        remaining=remaining,
        retry_after_s=retry_after,
        limit=limit,
        window_s=window_s,
    )
