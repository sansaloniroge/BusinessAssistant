from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from packages.shared.schemas.chat import ChatRequest, ChatResponse
from packages.shared.schemas.common import TenantContext

from apps.api.deps import get_chat_service, get_ctx
from apps.api.rate_limit import check_rate_limit
from apps.api.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    response: Response,
    ctx: TenantContext = Depends(get_ctx),
    svc: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    decision = check_rate_limit(
        tenant_id=str(ctx.tenant_id),
        user_id=str(ctx.user_id),
        route_key="chat",
    )
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)

    if not decision.allowed:
        response.headers["Retry-After"] = str(decision.retry_after_s)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return await svc.answer(ctx=ctx, req=req)
