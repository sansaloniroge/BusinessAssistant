import re

from packages.shared.schemas.common import Citation


class CitationService:
    CITATION_PATTERN = re.compile(r"\[C(\d+)]")

    # Guardrails V1
    MIN_STRICT_CITATIONS: int = 1
    MIN_STRICT_COVERAGE: float = 0.85  # ratio de tokens de cita válidos vs total tokens con patrón

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

    def validate_strict(
        self,
        answer: str,
        citations: list[Citation],
        *,
        retrieved_chunk_count: int | None = None,
    ) -> bool:
        """Valida guardrails de citas en modo strict.

        Reglas V1 (endurecidas):
        - Debe existir al menos un token [C#].
        - Debe existir al menos MIN_STRICT_CITATIONS citas construidas (citas válidas).
        - Validación de rango: 1..N donde N = retrieved_chunk_count (si se provee) o len(citations).
        - Cobertura mínima: (tokens [C#] cuyo # está en rango) / (total tokens [C#]) >= MIN_STRICT_COVERAGE.
        - Consistencia: cada índice referenciado en rango debe mapear a una Citation construida.
          (Evita pasar si el texto referencia [C2] pero build_citations no pudo construirla).
        """
        txt = answer or ""
        matches = list(self.CITATION_PATTERN.finditer(txt))
        if not matches:
            return False

        if not citations or len(citations) < self.MIN_STRICT_CITATIONS:
            return False

        # N: cuántos chunks estaban disponibles para citar
        n = int(retrieved_chunk_count or 0)
        if n <= 0:
            n = len(citations)

        # Índices tal como aparecen (para coverage)
        referenced_all = [int(m.group(1)) for m in matches]
        total_tokens = len(referenced_all)
        if total_tokens <= 0:
            return False

        def in_range(i: int) -> bool:
            return 1 <= i <= n

        valid_tokens = [i for i in referenced_all if in_range(i)]
        coverage = len(valid_tokens) / total_tokens
        if coverage < self.MIN_STRICT_COVERAGE:
            return False

        referenced_unique_in_range = set(valid_tokens)

        # Map de índices realmente construidos (por orden de aparición deduplicada)
        built_idxs = set(self.extract_citation_indices(txt))
        built_idxs_in_range = {i for i in built_idxs if in_range(i)}

        # Cada índice referenciado en rango debe estar en las citas construidas.
        # Nota: build_citations solo construye si idx está en rango y existe chunk.
        if not referenced_unique_in_range.issubset(built_idxs_in_range):
            return False

        # Exigir mínimo de citas construidas (ya validado arriba) y que al menos una se use.
        if not built_idxs_in_range:
            return False

        return True
