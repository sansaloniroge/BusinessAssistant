from __future__ import annotations

from uuid import UUID

from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.filters import FieldFilter, MetaFilters

from .ports import PermissionsService


class DefaultPermissionsService(PermissionsService):
    """Implementación mínima por defecto.

    - vector_filters_for: por ahora solo restringe por tenant (ya hay columna tenant_id) y
      permite filtrar por metadata. Se deja el hook para ACLs reales.
    - can_access_doc: en MVP devuelve True si el tenant coincide (la validación real suele
      requerir consultar una tabla de documentos/ACL).
    """

    async def vector_filters_for(self, ctx: TenantContext) -> MetaFilters:
        # Ejemplo de futuro: meta["allowed_groups"] = FieldFilter(op="$contains_any", value=ctx.scopes)
        return {}

    async def can_access_doc(self, *, ctx: TenantContext, doc_id: UUID) -> bool:
        # En este repo no existe aún una tabla de ACL/docs para validar.
        # El enforcement principal ahora mismo es a nivel vector store con tenant_id.
        return True

