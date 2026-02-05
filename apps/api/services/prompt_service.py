from packages.shared.schemas.chat import ChatMode


class PromptService:
    def system_prompt(self, mode: ChatMode) -> str:
        if mode == ChatMode.strict:
            return (
                "You must answer ONLY from the provided context.\n"
                "If the answer is not explicitly supported, respond: "
                "\"I don’t have enough evidence in the provided documents to answer that.\".\n"
                "Always include citations for factual statements. No citations => refuse.\n"
            )

        return (
            "You are an enterprise knowledge assistant.\n"
            "Answer using the provided context when possible.\n"
            "If context is insufficient, say you don't have enough information and ask for clarification.\n"
            "Always include citations for key claims.\n"
            "Never fabricate policies, procedures, numbers, or names.\n"
        )

    def build_context(self, retrieved_chunks) -> str:
        blocks: list[str] = []
        for i, c in enumerate(retrieved_chunks, start=1):
            section = c.metadata.get("section")
            page = c.metadata.get("page")
            blocks.append(
                f"[C{i}] doc_id={c.doc_id} title={c.title} chunk_id={c.chunk_id} "
                f"section={section} page={page}\n{c.content}\n"
            )
        return "\n---\n".join(blocks)
