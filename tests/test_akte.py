"""Tests für den lokalen Doku-Assistenten 'Theo Akte'."""

from theo.akte.dokumente import Abschnitt, zerteile
from theo.akte.suche import BM25Index, tokenisiere


def test_tokenisieren_entfernt_stoppwoerter_und_kleinschreibt():
    toks = tokenisiere("Die Kündigungsfrist ist und beträgt drei Monate")
    assert "die" not in toks and "ist" not in toks and "und" not in toks
    assert "kündigungsfrist" in toks
    assert "monate" in toks


def test_zerteilen_haelt_maximalgroesse_ein():
    text = "\n\n".join(f"Absatz {i} " + "x" * 100 for i in range(20))
    teile = zerteile(text, max_zeichen=300)
    assert all(len(t) <= 300 for t in teile)
    assert len(teile) > 1


def _index(abschnitte):
    ix = BM25Index()
    ix.bauen(abschnitte)
    return ix


def test_suche_findet_richtiges_dokument():
    abschnitte = [
        Abschnitt("miete.txt", 0, "Die Kündigungsfrist beträgt drei Monate zum Monatsende."),
        Abschnitt("urlaub.txt", 0, "Der Anspruch auf Erholungsurlaub beträgt 30 Arbeitstage."),
    ]
    treffer = _index(abschnitte).suche("Wie lang ist die Kündigungsfrist?", k=1)
    assert treffer and treffer[0].quelle == "miete.txt"


def test_suche_findet_zusammengesetzte_woerter():
    # "Urlaub" als Suchwort soll auch "Erholungsurlaub" finden.
    abschnitte = [
        Abschnitt("a.txt", 0, "Die Probezeit dauert sechs Monate."),
        Abschnitt("b.txt", 0, "Der Anspruch auf Erholungsurlaub beträgt 30 Tage."),
    ]
    treffer = _index(abschnitte).suche("Wie viel Urlaub gibt es?", k=1)
    assert treffer and treffer[0].quelle == "b.txt"


def test_speichern_und_laden(tmp_path):
    abschnitte = [Abschnitt("x.txt", 0, "Die Kaution beträgt 2850 Euro.")]
    ix = _index(abschnitte)
    pfad = tmp_path / "idx.json"
    ix.speichern(pfad)
    ix2 = BM25Index.laden(pfad)
    treffer = ix2.suche("Wie hoch ist die Kaution?", k=1)
    assert treffer and "Kaution" in treffer[0].text
