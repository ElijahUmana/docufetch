"""DocuFetch parser - extracts semantic structure from HTML documentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString


@dataclass
class CodeBlock:
    language: str
    code: str


@dataclass
class Section:
    url: str
    heading: str
    heading_level: int
    breadcrumb: list[str]
    text: str
    code_blocks: list[CodeBlock] = field(default_factory=list)


class Parser:
    """Parse an HTML doc page into a list of *Section* objects.

    Each section is anchored by a heading and contains the prose +
    code blocks that follow it, preserving the relationship between
    explanatory text and its associated code.
    """

    CONTENT_SELECTORS = [
        "main",
        "article",
        '[role="main"]',
        ".content",
        ".document",
        ".markdown-body",
        ".rst-content",
        ".doc-content",
        "#content",
        ".page-content",
        ".entry-content",
    ]

    HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    # ------------------------------------------------------------------

    def _find_content_root(self, soup: BeautifulSoup):
        for sel in self.CONTENT_SELECTORS:
            el = soup.select_one(sel)
            if el:
                return el
        return soup.body or soup

    @staticmethod
    def _code_language(element) -> str:
        """Best-effort language detection from class names."""
        for el in (element, element.parent):
            if el is None:
                continue
            classes = el.get("class", [])
            if isinstance(classes, str):
                classes = classes.split()
            for cls in classes:
                for prefix in ("language-", "highlight-", "sourceCode "):
                    if cls.startswith(prefix):
                        return cls[len(prefix):]
        return ""

    # ------------------------------------------------------------------

    def parse(self, url: str, html: str) -> list[Section]:
        soup = BeautifulSoup(html, "lxml")
        content = self._find_content_root(soup)

        sections: list[Section] = []
        cur_heading = (soup.title.string if soup.title else url) or url
        cur_level = 1
        breadcrumb: list[str] = [cur_heading]
        text_parts: list[str] = []
        code_blocks: list[CodeBlock] = []

        # Walk top-level children of the content root.  We use
        # recursive=False on a manual stack so that we don't double-
        # count text inside <pre> blocks.
        seen_pres: set[int] = set()

        for element in content.descendants:
            if isinstance(element, NavigableString):
                continue

            tag = element.name

            # --- headings ---
            if tag in self.HEADING_TAGS:
                # flush previous section
                joined = " ".join(text_parts).strip()
                if joined or code_blocks:
                    sections.append(
                        Section(
                            url=url,
                            heading=cur_heading,
                            heading_level=cur_level,
                            breadcrumb=list(breadcrumb),
                            text=joined,
                            code_blocks=list(code_blocks),
                        )
                    )
                cur_heading = element.get_text(strip=True)
                cur_level = int(tag[1])
                while len(breadcrumb) >= cur_level:
                    breadcrumb.pop()
                breadcrumb.append(cur_heading)
                text_parts = []
                code_blocks = []

            # --- code blocks ---
            elif tag == "pre":
                if id(element) in seen_pres:
                    continue
                seen_pres.add(id(element))
                code_el = element.find("code") or element
                code_text = code_el.get_text()
                lang = self._code_language(code_el) or self._code_language(element)
                code_blocks.append(CodeBlock(language=lang, code=code_text))

            # --- prose ---
            elif tag in ("p", "li", "dd", "td", "th", "blockquote", "dt"):
                # skip elements nested inside a <pre>
                if element.find_parent("pre"):
                    continue
                txt = element.get_text(strip=True)
                if txt and len(txt) > 3:
                    text_parts.append(txt)

        # flush last section
        joined = " ".join(text_parts).strip()
        if joined or code_blocks:
            sections.append(
                Section(
                    url=url,
                    heading=cur_heading,
                    heading_level=cur_level,
                    breadcrumb=list(breadcrumb),
                    text=joined,
                    code_blocks=list(code_blocks),
                )
            )

        return sections
