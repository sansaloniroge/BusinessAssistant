import os
from uuid import uuid4

import pytest

from apps.api.rate_limit import check_rate_limit


@pytest.mark.skipif(os.getenv("REDIS_URL") is None, reason="requiere REDIS_URL para test")
def test_rate_limit_blocks_after_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_USER", "2")
    monkeypatch.setenv("RATE_LIMIT_CHAT_WINDOW_S", "60")

    tenant_id = str(uuid4())
    user_id = str(uuid4())

    d1 = check_rate_limit(tenant_id=tenant_id, user_id=user_id, route_key="chat")
    assert d1.allowed is True

    d2 = check_rate_limit(tenant_id=tenant_id, user_id=user_id, route_key="chat")
    assert d2.allowed is True

    d3 = check_rate_limit(tenant_id=tenant_id, user_id=user_id, route_key="chat")
    assert d3.allowed is False
    assert d3.retry_after_s >= 0

