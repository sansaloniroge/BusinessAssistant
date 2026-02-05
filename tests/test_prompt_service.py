from uuid import uuid4

from apps.api.services.prompt_service import PromptService
from packages.shared.schemas.chat import ChatMode


class _Chunk:
    def __init__(self, *, doc_id, title, chunk_id, content, metadata=None):
        self.doc_id = doc_id
        self.title = title
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata or {}


def test_system_prompt_varies_by_mode():
    p = PromptService()
    strict = p.system_prompt(ChatMode.strict)
    normal = p.system_prompt(ChatMode.normal)

    assert "ONLY from the provided context" in strict
    assert "enterprise knowledge assistant" in normal


def test_build_context_numbers_citations_and_includes_metadata():
    p = PromptService()
    chunks = [
        _Chunk(
            doc_id=uuid4(),
            title="Doc",
            chunk_id="c1",
            content="Contenido",
            metadata={"section": "S", "page": 3},
        )
    ]

    ctx = p.build_context(chunks)
    assert "[C1]" in ctx
    assert "section=S" in ctx
    assert "page=3" in ctx
    assert "Contenido" in ctx
