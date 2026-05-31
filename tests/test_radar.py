"""Tests für Theo Radar (Erkennung + Tracking).

Diese Tests prüfen mit harten Zusicherungen, dass die Signalverarbeitung
physikalisch korrekte Werte liefert – unabhängig von evtl. gestörter
Konsolenausgabe.
"""

import numpy as np

from theo.radar.signal import erzeuge_signal
from theo.radar.szenario import RadarConfig, Ziel
from theo.radar.tracking import MultiTracker
from theo.radar.verarbeitung import finde_ziele, range_doppler_karte


def test_aufloesung_und_reichweite_plausibel():
    r = RadarConfig()
    assert abs(r.entfernung_aufloesung - 0.75) < 0.01
    assert 90 < r.entfernung_max < 100


def test_einzelziel_wird_korrekt_gemessen():
    r = RadarConfig()
    rng = np.random.default_rng(0)
    cube = erzeuge_signal(r, [Ziel(40.0, 5.0)], snr_db=25, rng=rng)
    P, ra, va = range_doppler_karte(cube, r)
    dets = finde_ziele(P, ra, va, pfa=1e-6)
    assert len(dets) >= 1
    bestes = max(dets, key=lambda d: d.snr_db)
    assert abs(bestes.entfernung - 40.0) < 1.0          # Entfernung stimmt
    assert abs(bestes.geschwindigkeit - 5.0) < 0.5      # Geschwindigkeit stimmt


def test_mehrere_ziele_getrennt_erkannt():
    r = RadarConfig()
    rng = np.random.default_rng(7)
    ziele = [Ziel(25.0, 4.0), Ziel(55.0, -6.0), Ziel(70.0, 0.0)]
    cube = erzeuge_signal(r, ziele, snr_db=22, rng=rng)
    P, ra, va = range_doppler_karte(cube, r)
    dets = finde_ziele(P, ra, va, pfa=1e-6)
    gemessen = sorted(d.entfernung for d in dets)
    # jedes wahre Ziel hat eine Detektion in der Nähe
    for z in ziele:
        assert any(abs(z.entfernung - g) < 1.0 for g in gemessen), z


def test_tracker_folgt_bewegtem_ziel():
    """Ein bewegtes Einzelziel muss korrekt verfolgt werden."""
    r = RadarConfig()
    rng = np.random.default_rng(3)
    tr = MultiTracker(dt=0.1)
    start, v = 30.0, 5.0
    for f in range(15):
        entf = start + v * 0.1 * f
        cube = erzeuge_signal(r, [Ziel(entf, v)], snr_db=25, rng=rng)
        P, ra, va = range_doppler_karte(cube, r)
        dets = finde_ziele(P, ra, va, pfa=1e-6)
        tracks = tr.schritt(dets)
    erwartet = start + v * 0.1 * 14
    assert len(tracks) == 1
    t = tracks[0]
    assert abs(t.entfernung - erwartet) < 2.0
    assert abs(t.geschwindigkeit - v) < 1.0


def test_tracker_haelt_zielanzahl_stabil():
    """Drei getrennte, bewegte Ziele -> genau drei bestätigte Tracks."""
    r = RadarConfig()
    rng = np.random.default_rng(5)
    tr = MultiTracker(dt=0.1)
    starts = [(20.0, 6.0), (75.0, -4.0), (50.0, 1.0)]
    tracks = []
    for f in range(15):
        cur = [Ziel(r0 + v * 0.1 * f, v) for r0, v in starts]
        cube = erzeuge_signal(r, cur, snr_db=22, rng=rng)
        P, ra, va = range_doppler_karte(cube, r)
        dets = finde_ziele(P, ra, va, pfa=1e-6)
        tracks = tr.schritt(dets)
    assert len(tracks) == 3
