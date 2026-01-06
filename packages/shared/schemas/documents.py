from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DocumentSourceType(str, Enum):
    upload = "upload"
    url = "url"
    confluence = "confluence"
    notion = "notion"


class AccessLevel(str, Enum):
    public = "public"
    restricted = "restricted"


class DocumentCreateRequest(BaseModel):
    title: str
    source_type: DocumentSourceType = DocumentSourceType.upload
    access_level: AccessLevel = AccessLevel.public
    tags: list[str] = Field(default_factory=list)
    department: str | None = None
    language: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: UUID
    title: str
    source_type: DocumentSourceType
    access_level: AccessLevel
    status: DocumentStatus
    tags: list[str] = Field(default_factory=list)
    department: str | None = None
    language: str | None = None
    blob_uri: str | None = None
    checksum: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentIngestRequest(BaseModel):
    document_id: UUID
    force_reindex: bool = False
