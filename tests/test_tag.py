"""Tests für den Tagesplaner 'Theo Tag'."""

from datetime import date, datetime

from theo.tag import modell
from theo.tag.erinnerung import faellige_erinnerungen, nachricht
from theo.tag.gewohnheiten import erkenne_gewohnheiten
from theo.tag.modell import Aktivitaet, Aufgabe, TagDaten, Termin
from theo.tag.planer import tagesplan


def test_gewohnheit_wird_erkannt():
    # Sport an drei Montagen (4./11./18.5.2026) gegen 7:00
    verlauf = [Aktivitaet("Sport", f"2026-05-{d}T07:05") for d in ("04", "11", "18")]
    gew = erkenne_gewohnheiten(verlauf)
    assert len(gew) == 1
    g = gew[0]
    assert g.name == "sport" and g.wochentag == 0  # Montag
    assert g.uhrzeit == "07:05"
    assert g.vorkommen == 3


def test_einmalige_aktivitaet_ist_keine_gewohnheit():
    verlauf = [Aktivitaet("Zahnarzt", "2026-05-04T10:00")]
    assert erkenne_gewohnheiten(verlauf) == []


def test_tagesplan_enthaelt_termin_gewohnheit_und_aufgabe():
    daten = TagDaten(
        termine=[Termin("Standup", "09:30", 15, [0, 1, 2, 3, 4])],
        aufgaben=[Aufgabe("Steuer", prioritaet=1, dauer_min=60)],
        verlauf=[Aktivitaet("Sport", f"2026-05-{d}T07:05") for d in ("04", "11", "18")],
    )
    plan = tagesplan(daten, date(2026, 6, 1))  # Montag
    arten = {e.art for e in plan}
    assert {"Termin", "Gewohnheit", "Aufgabe"} <= arten
    # nach Uhrzeit sortiert
    minuten = [e.minute for e in plan]
    assert minuten == sorted(minuten)


def test_aufgaben_kollidieren_nicht_mit_terminen():
    daten = TagDaten(
        termine=[Termin("Meeting", "09:00", 60)],
        aufgaben=[Aufgabe("Langaufgabe", dauer_min=120)],
    )
    plan = tagesplan(daten, date(2026, 6, 1))
    meeting = next(e for e in plan if e.art == "Termin")
    aufgabe = next(e for e in plan if e.art == "Aufgabe")
    # Aufgabe darf nicht in den Termin hineinragen
    ende_aufgabe = aufgabe.minute + aufgabe.dauer_min
    ueberlappt = aufgabe.minute < meeting.minute + meeting.dauer_min and ende_aufgabe > meeting.minute
    assert not ueberlappt


def test_erinnerung_ist_zur_richtigen_zeit_faellig():
    daten = TagDaten(termine=[Termin("Standup", "09:30", 15, erinnerung_min=5)])
    plan = tagesplan(daten, date(2026, 6, 1))
    # 5 Minuten vorher (09:25) -> fällig
    faellig = faellige_erinnerungen(plan, datetime(2026, 6, 1, 9, 25))
    assert len(faellig) == 1
    titel, text = nachricht(faellig[0])
    assert "Standup" in titel
    # zu einer anderen Zeit -> nicht fällig
    assert faellige_erinnerungen(plan, datetime(2026, 6, 1, 8, 0)) == []


def test_speichern_und_laden(tmp_path):
    pfad = tmp_path / "tag.json"
    daten = TagDaten(
        termine=[Termin("Sport", "07:00", 45, [0, 2, 4])],
        aufgaben=[Aufgabe("Lesen")],
        verlauf=[Aktivitaet("Kaffee", "2026-05-04T08:00")],
    )
    modell.speichern(daten, pfad)
    neu = modell.laden(pfad)
    assert neu.termine[0].titel == "Sport"
    assert neu.termine[0].wochentage == [0, 2, 4]
    assert neu.aufgaben[0].titel == "Lesen"
    assert neu.verlauf[0].name == "Kaffee"
