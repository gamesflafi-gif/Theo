"""Die Q&A-Engine: kombiniert Retrieval mit optionaler LLM-Antwort."""

from __future__ import annotations

from dataclasses import dataclass, field

import theo.qa.llm as llm
from theo.qa.retriever import Retriever, ScoredSection


@dataclass
class Answer:
    """Antwort der Engine inkl. Herkunft und genutzter Quellen."""

    question: str
    text: str
    sources: list[ScoredSection] = field(default_factory=list)
    used_llm: bool = False

    @property
    def source_titles(self) -> list[str]:
        return [f"{s.section.doc}: {s.section.heading_path}" for s in self.sources]


class QAEngine:
    """Beantwortet Football-Fragen.

    Standardmäßig (`mode="auto"`) wird die Claude API genutzt, falls verfügbar,
    sonst eine extraktive Antwort aus der Wissensbasis erzeugt.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        mode: str = "auto",
    ) -> None:
        if mode not in {"auto", "llm", "extractive"}:
            raise ValueError(f"Unbekannter Modus: {mode!r}")
        self.retriever = retriever or Retriever()
        self.mode = mode

    def _build_context(self, scored: list[ScoredSection]) -> str:
        blocks = []
        for s in scored:
            blocks.append(f"[{s.section.heading_path}]\n{s.section.text}")
        return "\n\n---\n\n".join(blocks)

    def _extractive_answer(self, question: str, scored: list[ScoredSection]) -> str:
        if not scored:
            return (
                "Dazu finde ich nichts in meiner Wissensbasis. Formuliere die Frage "
                "anders oder erweitere die Wissensdokumente unter "
                "`src/theo/knowledge/data/`."
            )
        best = scored[0].section
        lines = [
            f"Aus der Wissensbasis ({best.heading_path}):",
            "",
            best.body or best.text,
        ]
        if len(scored) > 1:
            lines.append("")
            lines.append("Weitere passende Abschnitte:")
            for s in scored[1:]:
                lines.append(f"  - {s.section.heading_path} ({s.section.doc})")
        return "\n".join(lines)

    def ask(self, question: str, *, top_k: int = 4) -> Answer:
        """Beantwortet eine Frage und liefert Text plus Quellen."""
        question = question.strip()
        scored = self.retriever.search(question, top_k=top_k)

        use_llm = self.mode == "llm" or (self.mode == "auto" and llm.is_available())
        if use_llm:
            try:
                context = self._build_context(scored)
                text = llm.answer_with_context(question, context)
                return Answer(question, text, sources=scored, used_llm=True)
            except Exception as exc:  # noqa: BLE001 – Fallback bei API-Fehlern
                if self.mode == "llm":
                    raise
                # Im Auto-Modus weich auf extraktiv zurückfallen.
                text = self._extractive_answer(question, scored)
                text += f"\n\n(Hinweis: LLM-Backend nicht nutzbar – {exc})"
                return Answer(question, text, sources=scored, used_llm=False)

        text = self._extractive_answer(question, scored)
        return Answer(question, text, sources=scored, used_llm=False)


__all__ = ["QAEngine", "Answer"]
