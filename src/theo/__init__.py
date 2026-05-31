"""Theo – eine KI rund um American Football.

Theo beantwortet Football-Fragen aus einer kuratierten Wissensbasis (optional
über die Claude API als RAG) und bietet eine pluggbare Pipeline für die Analyse
von Spiel- und Trainingsvideos.
"""

from theo.qa import Answer, QAEngine

__version__ = "0.1.0"

__all__ = ["QAEngine", "Answer", "__version__"]
