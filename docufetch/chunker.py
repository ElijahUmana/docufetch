"""DocuFetch chunker - semantic chunking that preserves code-context relationships.

Unlike naive text splitters that chunk by character count:
  * Never separates a code block from its preceding explanation
  * Embeds the heading hierarchy (breadcrumb) for retrieval context
  * Keeps section boundaries intact when possible
"""

from __future__ import annotations

from dataclasses import dataclass

from .parser import Section


@dataclass
class Chunk:
    content: str
    metadata: dict

    @property
    def token_estimate(self) -> int:
        return len(self.content.split())


class SemanticChunker:
    def __init__(self, max_chunk_tokens: int = 500, overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_section(section: Section) -> str:
        parts: list[str] = []

        if section.breadcrumb:
            parts.append(" > ".join(section.breadcrumb))
            parts.append("")

        if section.text:
            parts.append(section.text)
            parts.append("")

        for cb in section.code_blocks:
            lang_tag = f"```{cb.language}" if cb.language else "```"
            parts.append(lang_tag)
            parts.append(cb.code.strip())
            parts.append("```")
            parts.append("")

        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # Splitting large sections
    # ------------------------------------------------------------------

    def _split_large_section(self, section: Section) -> list[Chunk]:
        chunks: list[Chunk] = []
        ctx = (" > ".join(section.breadcrumb) + "\n\n") if section.breadcrumb else ""
        base_meta = {
            "url": section.url,
            "heading": section.heading,
            "heading_level": section.heading_level,
            "breadcrumb": " > ".join(section.breadcrumb),
        }

        if section.code_blocks:
            sentences = section.text.split(". ")
            n = len(section.code_blocks)
            per = max(1, len(sentences) // n)

            for i, cb in enumerate(section.code_blocks):
                start = i * per
                end = (start + per) if i < n - 1 else len(sentences)
                text_part = ". ".join(sentences[start:end])
                lang = f"```{cb.language}" if cb.language else "```"
                body = f"{ctx}{text_part}\n\n{lang}\n{cb.code.strip()}\n```"
                if body.strip():
                    chunks.append(Chunk(content=body, metadata={**base_meta, "has_code": True}))
        else:
            words = section.text.split()
            step = self.max_chunk_tokens - self.overlap_tokens
            for i in range(0, len(words), step):
                body = f"{ctx}{' '.join(words[i:i + self.max_chunk_tokens])}"
                if body.strip():
                    chunks.append(Chunk(content=body, metadata={**base_meta, "has_code": False}))

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_sections(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for section in sections:
            formatted = self._format_section(section)
            tok_est = len(formatted.split())

            if not section.text and not section.code_blocks:
                continue

            meta = {
                "url": section.url,
                "heading": section.heading,
                "heading_level": section.heading_level,
                "breadcrumb": " > ".join(section.breadcrumb),
                "has_code": len(section.code_blocks) > 0,
            }

            if tok_est <= self.max_chunk_tokens:
                chunks.append(Chunk(content=formatted, metadata=meta))
            else:
                chunks.extend(self._split_large_section(section))

        return chunks


# ------------------------------------------------------------------
# Naive chunker for comparison / demo
# ------------------------------------------------------------------


class NaiveChunker:
    """Character-count splitter that ignores semantic structure."""

    def __init__(self, chunk_size: int = 1500):
        self.chunk_size = chunk_size

    def chunk_text(self, text: str, url: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        for i in range(0, len(text), self.chunk_size):
            body = text[i : i + self.chunk_size]
            if body.strip():
                chunks.append(
                    Chunk(
                        content=body,
                        metadata={"url": url, "method": "naive", "has_code": "```" in body},
                    )
                )
        return chunks
