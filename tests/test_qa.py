import pytest

from theo.qa import QAEngine, Retriever


@pytest.fixture(scope="module")
def engine():
    # Extractive-Modus erzwingen: deterministisch und ohne Netz/Key.
    return QAEngine(mode="extractive")


def test_retriever_finds_touchdown():
    r = Retriever()
    results = r.search("Wie viele Punkte gibt ein Touchdown?")
    assert results
    top = results[0].section
    assert "punkte" in top.doc.lower() or "touchdown" in top.title.lower()


@pytest.mark.parametrize(
    "question,expect_term",
    [
        ("Was macht der Quarterback?", "quarterback"),
        ("Wie funktioniert ein Field Goal?", "field goal"),
        ("Was ist ein Blitz in der Defense?", "blitz"),
        ("Was passiert bei einer Safety?", "safety"),
        ("Welche Teams sind in der AFC West?", "broncos"),
        ("Was ist die GFL?", "german bowl"),
        ("Welche deutschen Teams spielen in der ELF?", "rhein fire"),
        ("Was ist das College Football Playoff?", "playoff"),
    ],
)
def test_engine_answers_are_relevant(engine, question, expect_term):
    answer = engine.ask(question)
    assert answer.text
    assert answer.sources, f"Keine Quellen für: {question}"
    combined = (answer.text + " " + " ".join(answer.source_titles)).lower()
    assert expect_term in combined


def test_unknown_question_degrades_gracefully(engine):
    answer = engine.ask("Wie koche ich Spaghetti carbonara?")
    assert answer.text  # Kein Crash, irgendeine Antwort.
    assert answer.used_llm is False


def test_mode_validation():
    with pytest.raises(ValueError):
        QAEngine(mode="bogus")
