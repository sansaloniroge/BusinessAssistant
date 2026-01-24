import pytest
from uuid import uuid4

from packages.shared.schemas.common import TenantContext


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(
        tenant_id=uuid4(),
        user_id=uuid4(),
        role="user",
        scopes=[],
    )

