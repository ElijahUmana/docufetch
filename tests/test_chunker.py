"""Tests for semantic chunking — demonstrating code-context preservation."""

from docufetch.parser import CodeBlock, Section
from docufetch.chunker import SemanticChunker


def test_code_stays_with_explanation():
    section = Section(
        url="https://docs.example.com/quickstart",
        heading="Making a GET request",
        heading_level=2,
        breadcrumb=["Quickstart", "Making a GET request"],
        text=(
            "To make a GET request, use the get() function. "
            "It sends an HTTP GET and returns a Response object."
        ),
        code_blocks=[
            CodeBlock(
                language="python",
                code="import httpx\nresponse = httpx.get('https://api.example.com')\nprint(response.json())",
            )
        ],
    )

    chunks = SemanticChunker(max_chunk_tokens=500).chunk_sections([section])

    assert len(chunks) == 1
    assert "GET request" in chunks[0].content
    assert "httpx.get(" in chunks[0].content
    assert chunks[0].metadata["has_code"] is True


def test_heading_hierarchy_preserved():
    section = Section(
        url="https://docs.example.com/auth",
        heading="Bearer Tokens",
        heading_level=3,
        breadcrumb=["Authentication", "Token-based", "Bearer Tokens"],
        text="Pass your token in the Authorization header.",
        code_blocks=[
            CodeBlock(language="python", code='headers = {"Authorization": "Bearer sk-xxx"}')
        ],
    )

    chunks = SemanticChunker().chunk_sections([section])

    assert len(chunks) == 1
    assert "Authentication > Token-based > Bearer Tokens" in chunks[0].content


def test_empty_sections_excluded():
    section = Section(
        url="https://docs.example.com/empty",
        heading="Empty",
        heading_level=2,
        breadcrumb=["Empty"],
        text="",
        code_blocks=[],
    )

    assert SemanticChunker().chunk_sections([section]) == []


def test_large_section_splits_preserve_code():
    section = Section(
        url="https://docs.example.com/large",
        heading="Large Section",
        heading_level=2,
        breadcrumb=["Large Section"],
        text=" ".join(["word"] * 1000),
        code_blocks=[
            CodeBlock(language="python", code="print('hello')"),
            CodeBlock(language="python", code="print('world')"),
        ],
    )

    chunks = SemanticChunker(max_chunk_tokens=200).chunk_sections([section])

    for chunk in chunks:
        if chunk.metadata.get("has_code"):
            assert "```" in chunk.content
