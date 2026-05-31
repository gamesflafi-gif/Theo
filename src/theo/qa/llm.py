"""Optionales Claude-Backend für RAG-Antworten.

Wird nur aktiv, wenn das Paket `anthropic` installiert und ein API-Key gesetzt
ist (ANTHROPIC_API_KEY). Andernfalls fällt Theo auf extraktive Antworten zurück.
"""

from __future__ import annotations

import os

# Aktuelles, leistungsfähiges Standardmodell für die Q&A.
DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = (
    "Du bist Theo, ein Experte für American Football. Beantworte die Frage des "
    "Nutzers klar, korrekt und auf Deutsch. Stütze dich vorrangig auf den "
    "bereitgestellten KONTEXT aus der Wissensbasis. Wenn der Kontext nicht "
    "ausreicht, ergänze vorsichtig mit allgemeinem Football-Wissen und kennzeichne "
    "das. Erfinde keine Regeln oder Statistiken."
)


def is_available() -> bool:
    """True, wenn ein Claude-Backend genutzt werden kann."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def answer_with_context(
    question: str,
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Erzeugt eine RAG-Antwort über die Claude API.

    Raises:
        RuntimeError: wenn das Backend nicht verfügbar ist.
    """
    if not is_available():
        raise RuntimeError(
            "Claude-Backend nicht verfügbar: ANTHROPIC_API_KEY setzen und "
            "`pip install theo[llm]` ausführen."
        )

    import anthropic

    client = anthropic.Anthropic()
    user_content = (
        f"KONTEXT (Auszüge aus der Wissensbasis):\n{context}\n\n"
        f"FRAGE: {question}"
    )
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    parts = [block.text for block in message.content if block.type == "text"]
    return "\n".join(parts).strip()


__all__ = ["is_available", "answer_with_context", "DEFAULT_MODEL"]
