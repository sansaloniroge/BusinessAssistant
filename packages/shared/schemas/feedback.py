from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    run_id: UUID
    rating: int = Field(ge=-1, le=1)  # -1 or +1
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    rating: int
    comment: str | None
    created_at: datetime
