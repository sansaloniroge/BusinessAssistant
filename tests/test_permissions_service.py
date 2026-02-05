import pytest
from uuid import uuid4

from apps.api.services.permissions import DefaultPermissionsService
from packages.shared.schemas.common import TenantContext


@pytest.mark.asyncio
async def test_default_permissions_service_vector_filters_and_access():
    svc = DefaultPermissionsService()
    ctx = TenantContext(tenant_id=str(uuid4()), user_id=uuid4(), role="user", scopes=[])

    filters = await svc.vector_filters_for(ctx)
    # By default, it does not add restrictions (isolation is by tenant_id in the vector store).
    assert isinstance(filters, dict)
    assert filters == {}

    can = await svc.can_access_doc(ctx=ctx, doc_id=uuid4())
    assert can is True

