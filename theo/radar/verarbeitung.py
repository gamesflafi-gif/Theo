"""Range-Doppler-Verarbeitung und CFAR-Zielerkennung.

Schritt 1 (Range-Doppler-Karte): Eine 2D-FFT des Datenwürfels macht jedes Ziel
als hellen Punkt sichtbar – die eine Achse ist die Entfernung, die andere die
Geschwindigkeit.

Schritt 2 (CFAR = Constant False Alarm Rate): Der Standard-Algorithmus echter
Radare. Statt einer festen Schwelle schätzt er das Rauschen *lokal* aus den
Nachbarzellen und erkennt nur, was deutlich darüber liegt. So bleibt die
Fehlalarmrate konstant, egal wie stark das Hintergrundrauschen schwankt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .szenario import C, RadarConfig


@dataclass
class Detektion:
    entfernung: float          # [m]
    geschwindigkeit: float     # [m/s]
    snr_db: float
    range_bin: int
    dopp_bin: int


def range_doppler_karte(cube: np.ndarray, radar: RadarConfig):
    """Wandelt den Datenwürfel in eine Range-Doppler-Leistungskarte.

    Rückgabe: (leistung[n_range, n_dopp], entfernungs_achse, geschw_achse)
    """
    n_c, n_s = cube.shape
    # Fensterung gegen Nebenkeulen (Hann), je Achse
    fenster = np.hanning(n_s)[None, :] * np.hanning(n_c)[:, None]
    x = cube * fenster

    # Range-FFT entlang fast time, nur positive Entfernungen behalten
    bereich = np.fft.fft(x, axis=1)[:, : n_s // 2]
    # Doppler-FFT entlang slow time, Nullgeschwindigkeit in die Mitte schieben
    rd = np.fft.fftshift(np.fft.fft(bereich, axis=0), axes=0)
    leistung = np.abs(rd) ** 2  # Form: (n_dopp, n_range)
    leistung = leistung.T       # -> (n_range, n_dopp)

    n_range = leistung.shape[0]
    entfernung_achse = np.arange(n_range) * radar.entfernung_aufloesung
    d = np.arange(n_c) - n_c // 2
    geschw_achse = d * radar.geschw_aufloesung
    return leistung, entfernung_achse, geschw_achse


def _integralbild(p: np.ndarray) -> np.ndarray:
    """Summierte Flächentabelle für schnelle Fenstersummen (mit Nullrand)."""
    s = np.cumsum(np.cumsum(p, axis=0), axis=1)
    return np.pad(s, ((1, 0), (1, 0)))


def ca_cfar(leistung: np.ndarray, n_train: int = 8, n_guard: int = 2,
            pfa: float = 1e-6) -> np.ndarray:
    """Zell-mittelnde CFAR-Detektion (2D). Gibt eine boolesche Trefferkarte zurück.

    Für jede Zelle wird das Rauschen aus einem Ring von Trainingszellen geschätzt
    (Schutzzellen direkt um die Zelle herum bleiben außen vor).
    """
    fenster = n_train + n_guard
    integ = _integralbild(leistung)
    H, W = leistung.shape

    def block_summe(halb: int) -> np.ndarray:
        """Summe eines (2*halb+1)-Quadrats um jede Zelle (Rand = abgeschnitten)."""
        summe = np.zeros_like(leistung)
        rr = np.arange(H)
        cc = np.arange(W)
        r0 = np.clip(rr - halb, 0, H)[:, None]
        r1 = np.clip(rr + halb + 1, 0, H)[:, None]
        c0 = np.clip(cc - halb, 0, W)[None, :]
        c1 = np.clip(cc + halb + 1, 0, W)[None, :]
        summe = (integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0])
        anzahl = (r1 - r0) * (c1 - c0)
        return summe, anzahl

    s_aussen, n_aussen = block_summe(fenster)
    s_innen, n_innen = block_summe(n_guard)
    train_summe = s_aussen - s_innen
    train_anzahl = np.maximum(n_aussen - n_innen, 1)
    rauschen = train_summe / train_anzahl

    # Schwellenfaktor aus gewünschter Fehlalarmrate
    alpha = train_anzahl * (pfa ** (-1.0 / train_anzahl) - 1.0)
    schwelle = alpha * rauschen
    return leistung > schwelle


def finde_ziele(leistung, entfernung_achse, geschw_achse,
                n_train=8, n_guard=2, pfa=1e-6, min_abstand=2) -> list[Detektion]:
    """CFAR + Non-Maximum-Suppression -> eine Detektion pro Ziel."""
    treffer = ca_cfar(leistung, n_train, n_guard, pfa)
    rauschboden = np.median(leistung)
    kandidaten = [
        (leistung[r, c], r, c)
        for r, c in zip(*np.where(treffer))
    ]
    kandidaten.sort(reverse=True)  # stärkste zuerst

    belegt = np.zeros_like(treffer, dtype=bool)
    detektionen: list[Detektion] = []
    for lst, r, c in kandidaten:
        if belegt[r, c]:
            continue
        snr = 10 * np.log10(lst / max(rauschboden, 1e-12))
        detektionen.append(Detektion(
            entfernung=float(entfernung_achse[r]),
            geschwindigkeit=float(geschw_achse[c]),
            snr_db=float(snr),
            range_bin=int(r), dopp_bin=int(c),
        ))
        # Umgebung sperren (Non-Maximum-Suppression)
        r0, r1 = max(0, r - min_abstand), min(leistung.shape[0], r + min_abstand + 1)
        c0, c1 = max(0, c - min_abstand), min(leistung.shape[1], c + min_abstand + 1)
        belegt[r0:r1, c0:c1] = True
    return detektionen
