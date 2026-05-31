from theo.knowledge import load_sections


def test_sections_load():
    sections = load_sections()
    assert len(sections) > 20
    # Jeder Abschnitt hat Titel und Text.
    for sec in sections:
        assert sec.title
        assert sec.text
        assert sec.doc.endswith(".md")


def test_known_topics_present():
    corpus = " ".join(s.text.lower() for s in load_sections())
    for term in ["quarterback", "touchdown", "field goal", "blitz", "safety"]:
        assert term in corpus, f"Thema fehlt in der Wissensbasis: {term}"


def test_heading_path_nesting():
    # Mindestens ein Abschnitt sollte verschachtelte Überschriften haben.
    assert any(">" in s.heading_path for s in load_sections())
