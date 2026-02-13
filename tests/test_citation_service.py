from uuid import uuid4

from apps.api.services.citation_service import CitationService
from packages.shared.schemas.common import Citation


class _Chunk:
    def __init__(self, *, doc_id, title, chunk_id, content, score=0.5, metadata=None):
        self.doc_id = doc_id
        self.title = title
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


def test_extract_citation_indices_dedup_and_order():
    s = CitationService()
    ans = "Texto [C2] y luego [C1] y repetido [C2]."
    assert s.extract_citation_indices(ans) == [2, 1]


def test_build_citations_maps_to_retrieved_chunks_and_snippet_truncates():
    s = CitationService()

    long_text = "a" * 350
    chunks = [
        _Chunk(doc_id=uuid4(), title="Doc A", chunk_id="c1", content=long_text, metadata={"page": 1}),
        _Chunk(doc_id=uuid4(), title="Doc B", chunk_id="c2", content="short", metadata={"section": "S1"}),
    ]

    ans = "Soporta A [C1] y B [C2]."
    citations = s.build_citations(chunks, ans)

    assert len(citations) == 2
    assert isinstance(citations[0], Citation)
    assert citations[0].chunk_id == "c1"
    assert citations[0].page == 1
    assert citations[0].snippet.endswith("…")
    assert len(citations[0].snippet) == 301

    assert citations[1].chunk_id == "c2"
    assert citations[1].section == "S1"
    assert citations[1].snippet == "short"


def test_validate_strict_requires_at_least_one_valid_citation():
    s = CitationService()
    assert s.validate_strict("Sin citas", []) is False

    any_citation = Citation(
        doc_id=uuid4(),
        title="Doc",
        chunk_id="c1",
        snippet="x",
        section=None,
        page=None,
        score=0.1,
        metadata={},
    )
    assert s.validate_strict("Hay [C1]", [any_citation]) is True


def test_validate_strict_rejects_out_of_range_index_when_chunk_count_provided():
    s = CitationService()
    any_citation = Citation(
        doc_id=uuid4(),
        title="Doc",
        chunk_id="c1",
        snippet="x",
        section=None,
        page=None,
        score=0.1,
        metadata={},
    )

    # Solo había 2 chunks recuperados; [C999] debe forzar rechazo (coverage muy baja)
    assert s.validate_strict("Dato [C1] y ruido [C999]", [any_citation], retrieved_chunk_count=2) is False


def test_validate_strict_rejects_mixed_valid_and_invalid_below_coverage_threshold():
    s = CitationService()
    any_citation = Citation(
        doc_id=uuid4(),
        title="Doc",
        chunk_id="c1",
        snippet="x",
        section=None,
        page=None,
        score=0.1,
        metadata={},
    )

    # 1 válida sobre 3 tokens => 0.33 < 0.85
    assert s.validate_strict("A [C1] B [C999] C [C999]", [any_citation], retrieved_chunk_count=2) is False
