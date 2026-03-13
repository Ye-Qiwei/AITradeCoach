"""Local fallback for markdown.markdown API using markdown-it-py."""

from __future__ import annotations

from markdown_it import MarkdownIt


def markdown(text: str, extensions: list[str] | None = None) -> str:
    _ = extensions
    return MarkdownIt("commonmark", {"html": True}).enable("table").render(text)
