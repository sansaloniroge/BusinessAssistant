import re

from packages.shared.schemas.common import Citation


class CitationService:
    CITATION_PATTERN = re.compile(r"\[C(\d+)\]")

    def extract_citation_indices(self, answer: str) -> list[int]:
        indices: list[int] = []
        for m in self.CITATION_PATTERN.finditer(answer):
            idx = int(m.group(1))
            if idx not in indices:
                indices.append(idx)
        return indices

    def build_citations(self, retrieved_chunks, answer: str) -> list[Citation]:
        idxs = self.extract_citation_indices(answer)
        citations: list[Citation] = []

        for idx in idxs:
            if 1 <= idx <= len(retrieved_chunks):
                c = retrieved_chunks[idx - 1]
                snippet = (c.content[:300] + "…") if len(c.content) > 300 else c.content
                citations.append(
                    Citation(
                        doc_id=c.doc_id,
                        title=c.title,
                        chunk_id=c.chunk_id,
                        snippet=snippet,
                        section=c.metadata.get("section"),
                        page=c.metadata.get("page"),
                        score=c.score,
                        metadata=c.metadata,
                    )
                )
        return citations

    def validate_strict(self, answer: str, citations: list[Citation]) -> bool:
        # MVP strict rule: must include at least one [C#]
        return bool(self.CITATION_PATTERN.search(answer)) and bool(citations)
