"""Lädt die Football-Wissensbasis und zerlegt sie in durchsuchbare Abschnitte."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

_DATA_PACKAGE = "theo.knowledge.data"


@dataclass(frozen=True)
class Section:
    """Ein Abschnitt der Wissensbasis (eine Überschrift mit ihrem Text)."""

    doc: str          # Quelldatei, z. B. "03_positionen_offense.md"
    title: str        # Überschriftentext, z. B. "Quarterback (QB)"
    heading_path: str # Pfad der Überschriften, z. B. "Positionen … > Quarterback (QB)"
    text: str         # Voller Text des Abschnitts (inkl. Überschrift)

    @property
    def body(self) -> str:
        """Abschnittstext ohne die Überschriftenzeile."""
        lines = self.text.splitlines()
        return "\n".join(lines[1:]).strip() if lines else ""


def _split_into_sections(doc_name: str, content: str) -> list[Section]:
    """Zerlegt ein Markdown-Dokument an seinen Überschriften (#, ##, ###)."""
    sections: list[Section] = []
    # Stack der aktuellen Überschriften je Ebene für den heading_path.
    stack: list[tuple[int, str]] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    current_title: str | None = None
    current_path: str = ""
    buffer: list[str] = []

    def flush() -> None:
        if current_title is None:
            return
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                Section(
                    doc=doc_name,
                    title=current_title,
                    heading_path=current_path,
                    text=text,
                )
            )

    for line in content.splitlines():
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Stack auf die aktuelle Ebene zurückschneiden.
            stack[:] = [(lvl, t) for (lvl, t) in stack if lvl < level]
            stack.append((level, title))
            current_title = title
            current_path = " > ".join(t for _, t in stack)
            buffer = [line]
        else:
            buffer.append(line)

    flush()
    return sections


@lru_cache(maxsize=1)
def load_sections() -> tuple[Section, ...]:
    """Lädt alle Abschnitte der gebündelten Wissensbasis (gecacht)."""
    sections: list[Section] = []
    data_root = resources.files(_DATA_PACKAGE)
    for entry in sorted(data_root.iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".md"):
            content = entry.read_text(encoding="utf-8")
            sections.extend(_split_into_sections(entry.name, content))
    return tuple(sections)


def knowledge_dir() -> Path:
    """Pfad zum Verzeichnis der Wissensdokumente (für Tooling/Tests)."""
    return Path(__file__).parent / "data"


__all__ = ["Section", "load_sections", "knowledge_dir"]
