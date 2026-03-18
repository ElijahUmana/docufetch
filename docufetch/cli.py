"""DocuFetch CLI - command-line interface."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """DocuFetch - Autonomous documentation crawler & vectorizer for LLM context grounding."""


# ======================================================================
# crawl
# ======================================================================


@cli.command()
@click.argument("url")
@click.option("--max-pages", "-m", default=30, help="Maximum pages to crawl")
@click.option("--chunk-size", "-c", default=500, help="Target chunk size (tokens)")
@click.option("--name", "-n", default=None, help="Name for this doc source")
@click.option("--delay", "-d", default=0.3, type=float, help="Delay between requests (s)")
def crawl(url: str, max_pages: int, chunk_size: int, name: str | None, delay: float):
    """Crawl and index documentation from URL."""
    from urllib.parse import urlparse

    from .chunker import SemanticChunker
    from .crawler import Crawler
    from .parser import Parser
    from .store import VectorStore

    source_name = name or urlparse(url).netloc

    console.print(f"\n[bold blue]DocuFetch[/bold blue] — Indexing [cyan]{url}[/cyan]\n")

    # 1 — Crawl
    console.print("[bold]Phase 1:[/bold] Crawling documentation...")
    crawler = Crawler(url, max_pages=max_pages, delay=delay)
    pages = crawler.crawl()
    console.print(f"  [green]✓[/green] Crawled {len(pages)} pages\n")

    if not pages:
        console.print("[red]No pages found — check the URL.[/red]")
        return

    # 2 — Parse
    console.print("[bold]Phase 2:[/bold] Parsing semantic structure...")
    parser = Parser()
    all_sections = []
    for page_url, html in pages:
        all_sections.extend(parser.parse(page_url, html))
    code_sections = sum(1 for s in all_sections if s.code_blocks)
    console.print(
        f"  [green]✓[/green] {len(all_sections)} sections "
        f"({code_sections} with code blocks)\n"
    )

    # 3 — Chunk
    console.print("[bold]Phase 3:[/bold] Semantic chunking (preserving code-context pairs)...")
    chunker = SemanticChunker(max_chunk_tokens=chunk_size)
    chunks = chunker.chunk_sections(all_sections)
    code_chunks = sum(1 for c in chunks if c.metadata.get("has_code"))
    console.print(
        f"  [green]✓[/green] {len(chunks)} semantic chunks "
        f"({code_chunks} with embedded code)\n"
    )

    # 4 — Index
    console.print("[bold]Phase 4:[/bold] Vectorizing and indexing...")
    store = VectorStore()
    n = store.index(source_name, chunks)
    console.print(f"  [green]✓[/green] Indexed {n} vectors\n")

    console.print(
        Panel(
            f"[bold green]Done![/bold green]\n\n"
            f"Source : {source_name}\n"
            f"Pages  : {len(pages)}\n"
            f"Chunks : {len(chunks)} ({code_chunks} with code)\n\n"
            f"Query with:\n  [cyan]docufetch query \"{source_name}\" \"your question\"[/cyan]",
            title="Indexing complete",
            border_style="green",
        )
    )


# ======================================================================
# query
# ======================================================================


@cli.command()
@click.argument("source")
@click.argument("question")
@click.option("--n-results", "-n", default=5, help="Number of results")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def query(source: str, question: str, n_results: int, json_output: bool):
    """Query indexed documentation."""
    from .store import VectorStore

    store = VectorStore()
    results = store.query(source, question, n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if json_output:
        out = [
            {
                "rank": i + 1,
                "relevance": round(1 - d, 4),
                "url": m.get("url", ""),
                "heading": m.get("heading", ""),
                "content": doc,
            }
            for i, (doc, m, d) in enumerate(zip(docs, metas, dists))
        ]
        click.echo(json.dumps(out, indent=2))
        return

    console.print(
        f"\n[bold blue]DocuFetch[/bold blue] — Results for: [cyan]{question}[/cyan]\n"
    )

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        rel = 1 - dist
        color = "green" if rel > 0.5 else "yellow" if rel > 0.3 else "red"
        console.print(
            Panel(
                doc,
                title=f"[{color}]#{i + 1}[/{color}] {meta.get('heading', '')} — relevance {rel:.1%}",
                subtitle=f"[dim]{meta.get('url', '')}[/dim]",
                border_style=color,
                padding=(1, 2),
            )
        )
        console.print()


# ======================================================================
# compare  (demo: semantic vs naive chunking)
# ======================================================================


@cli.command()
@click.argument("url")
def compare(url: str):
    """Compare semantic vs naive chunking on a single page (demo)."""
    import requests
    from .chunker import NaiveChunker, SemanticChunker
    from .parser import Parser

    console.print(f"\n[bold blue]DocuFetch[/bold blue] — Chunking comparison\n")

    resp = requests.get(url, timeout=10)
    html = resp.text
    plain = " ".join(html.split())  # flatten for naive

    # Naive
    naive = NaiveChunker(chunk_size=1200)
    naive_chunks = naive.chunk_text(plain, url)

    # Semantic
    parser = Parser()
    sections = parser.parse(url, html)
    sem = SemanticChunker(max_chunk_tokens=300)
    sem_chunks = sem.chunk_sections(sections)

    # Display
    console.print(f"[bold red]Naive chunking[/bold red]: {len(naive_chunks)} chunks (fixed 1200-char splits)\n")
    for i, c in enumerate(naive_chunks[:3]):
        console.print(
            Panel(
                c.content[:600] + ("..." if len(c.content) > 600 else ""),
                title=f"Naive chunk {i + 1}",
                border_style="red",
                padding=(1, 2),
            )
        )

    console.print(f"\n[bold green]Semantic chunking[/bold green]: {len(sem_chunks)} chunks (structure-aware)\n")
    code_shown = 0
    for i, c in enumerate(sem_chunks):
        if c.metadata.get("has_code") and code_shown < 3:
            console.print(
                Panel(
                    c.content[:600] + ("..." if len(c.content) > 600 else ""),
                    title=f"Semantic chunk {i + 1} — {c.metadata.get('heading', '')}",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            code_shown += 1


# ======================================================================
# list / inspect / clear
# ======================================================================


@cli.command(name="list")
def list_sources():
    """List all indexed documentation sources."""
    from .store import VectorStore

    store = VectorStore()
    sources = store.list_sources()

    if not sources:
        console.print("[yellow]No documentation indexed yet.[/yellow]")
        return

    table = Table(title="Indexed Sources")
    table.add_column("Collection", style="cyan")
    table.add_column("Source", style="green")
    for col_name, source in sources:
        table.add_row(col_name, source)
    console.print(table)


@cli.command()
@click.argument("source")
@click.option("--n-samples", "-n", default=3, help="Number of sample chunks to show")
def inspect(source: str, n_samples: int):
    """Show sample chunks for a documentation source."""
    from .store import VectorStore

    store = VectorStore()
    collection = store.get_collection(source)

    console.print(
        f"\n[bold blue]DocuFetch[/bold blue] — Chunks from [cyan]{source}[/cyan]\n"
    )
    console.print(f"Total chunks: [green]{collection.count()}[/green]\n")

    peek = collection.peek(n_samples)
    for i, (doc, meta) in enumerate(zip(peek["documents"], peek["metadatas"])):
        tag = "[green]code[/green]" if meta.get("has_code") else "[dim]text[/dim]"
        console.print(
            Panel(
                doc[:500] + ("..." if len(doc) > 500 else ""),
                title=f"Chunk {i + 1} [{tag}] — {meta.get('heading', '')}",
                subtitle=f"[dim]{meta.get('breadcrumb', '')}[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()


@cli.command()
@click.argument("source", required=False)
@click.option("--all", "clear_all", is_flag=True, help="Clear everything")
def clear(source: str | None, clear_all: bool):
    """Clear indexed documentation."""
    from .store import VectorStore

    store = VectorStore()
    if clear_all:
        store.clear()
        console.print("[green]All indexed data cleared.[/green]")
    elif source:
        store.delete(source)
        console.print(f"[green]Cleared: {source}[/green]")
    else:
        console.print("[red]Specify a source or use --all[/red]")


# ======================================================================
# entry point
# ======================================================================


def main():
    cli()


if __name__ == "__main__":
    main()
