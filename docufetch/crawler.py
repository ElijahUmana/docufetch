"""DocuFetch crawler - fetches documentation pages from a given root URL."""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class Crawler:
    """Crawl documentation sites starting from a root URL.

    Stays within the same domain and path prefix to avoid
    wandering off into unrelated pages.
    """

    SKIP_EXTENSIONS = frozenset([
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
        ".pdf", ".zip", ".tar", ".gz",
    ])

    def __init__(self, base_url: str, max_pages: int = 50, delay: float = 0.3):
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.visited: set[str] = set()
        self._parsed_base = urlparse(base_url)
        self._base_path = self._parsed_base.path.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self._parsed_base.netloc:
            return False
        if self._base_path and not parsed.path.startswith(self._base_path):
            return False
        if any(parsed.path.lower().endswith(ext) for ext in self.SKIP_EXTENSIONS):
            return False
        return True

    @staticmethod
    def _normalize(url: str) -> str:
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme}://{p.netloc}{path}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> list[tuple[str, str]]:
        """Return a list of (url, html) for every crawled page."""
        queue: deque[str] = deque([self.base_url])
        results: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Crawling...", total=None)

            while queue and len(results) < self.max_pages:
                url = queue.popleft()
                norm = self._normalize(url)

                if norm in self.visited:
                    continue
                self.visited.add(norm)

                try:
                    progress.update(
                        task,
                        description=f"[{len(results) + 1}] {norm[:80]}",
                    )
                    resp = requests.get(
                        norm,
                        timeout=10,
                        headers={"User-Agent": "DocuFetch/1.0 (documentation indexer)"},
                    )
                    if resp.status_code != 200:
                        continue
                    content_type = resp.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        continue

                    html = resp.text
                    results.append((norm, html))

                    # Discover links
                    soup = BeautifulSoup(html, "lxml")
                    for anchor in soup.find_all("a", href=True):
                        link = self._normalize(urljoin(norm, anchor["href"]))
                        if self._is_valid_url(link) and link not in self.visited:
                            queue.append(link)

                    time.sleep(self.delay)
                except Exception as exc:
                    console.print(f"[yellow]Skipped {norm}: {exc}[/yellow]")

        return results
