# DocuFetch

Autonomous documentation crawler and vectorizer for LLM context grounding.

DocuFetch crawls software documentation sites, parses them into semantically meaningful chunks that preserve the relationship between code and explanation, and stores them in a local vector database for instant retrieval — purpose-built for RAG pipelines.

## The Problem

LLMs generate deprecated or incorrect code because their training data is stale. The standard fix is RAG (Retrieval-Augmented Generation), but existing tools either:

- **Enterprise IDE extensions** (Copilot, Cursor): proprietary, lag behind updates, require subscriptions
- **Generic web scrapers** (BeautifulSoup): split code blocks in half, orphan code from explanations, chunk raw HTML noise

## The Solution: Semantic Chunking

DocuFetch parses documentation HTML **structurally** — it understands headings, paragraphs, and code blocks. Unlike naive character-count splitters:

- Code blocks are **never separated** from their explanatory text
- Heading hierarchy is embedded as breadcrumbs for retrieval context
- Section boundaries are preserved
- No HTML/CSS/JS noise in chunks

## Installation

```bash
git clone https://github.com/ElijahUmana/docufetch.git
cd docufetch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Crawl & index documentation

```bash
docufetch crawl "https://www.python-httpx.org/" --max-pages 30
```

Output:
```
Phase 1: Crawling documentation...
  ✓ Crawled 15 pages

Phase 2: Parsing semantic structure...
  ✓ 123 sections (80 with code blocks)

Phase 3: Semantic chunking (preserving code-context pairs)...
  ✓ 128 semantic chunks (85 with embedded code)

Phase 4: Vectorizing and indexing...
  ✓ Indexed 128 vectors
```

### Query indexed docs

```bash
docufetch query "www.python-httpx.org" "how to make async HTTP requests"
```

Returns ranked results with relevance scores, each containing the explanation and code together:

```
#1 Making Async requests — relevance 34.5%
────────────────────────────────────────────
Async Support > Making Async requests

To make asynchronous requests, you'll need an AsyncClient.

>>> async with httpx.AsyncClient() as client:
...     r = await client.get('https://www.example.com/')
```

### Compare chunking strategies

```bash
docufetch compare "https://www.python-httpx.org/quickstart/"
```

Side-by-side comparison showing naive chunking (60 chunks of raw HTML) vs semantic chunking (19 clean, structured chunks).

### Other commands

```bash
docufetch list                    # List indexed sources
docufetch inspect <source>        # View sample chunks
docufetch clear --all             # Clear all indexed data
```

## Architecture

```
URL → Crawler → Parser → Semantic Chunker → Vector Store → Query
```

| Component | Technology |
|-----------|-----------|
| Crawling | `requests` + `BeautifulSoup` |
| Parsing | Semantic HTML structural analysis |
| Chunking | Custom structure-aware chunker |
| Embeddings | `all-MiniLM-L6-v2` via ChromaDB |
| Vector Store | ChromaDB (local, persistent) |
| CLI | Click + Rich |

## Tech Stack

- Python 3.10+
- BeautifulSoup4 + lxml
- ChromaDB
- Click
- Rich
