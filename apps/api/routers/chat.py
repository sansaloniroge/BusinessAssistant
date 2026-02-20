from __future__ import annotations

from fastapi import APIRouter, Depends

from packages.shared.schemas.chat import ChatRequest, ChatResponse
from packages.shared.schemas.common import TenantContext

from apps.api.deps import get_chat_service, get_ctx
from apps.api.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    ctx: TenantContext = Depends(get_ctx),
    svc: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await svc.answer(ctx=ctx, req=req)
