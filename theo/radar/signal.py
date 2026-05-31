"""Aus Zielen ein realistisches FMCW-Empfangssignal erzeugen.

Ein FMCW-Radar sendet einen Frequenz-Rampen-Ton ("Chirp") und mischt das Echo
mit dem Sendesignal. Daraus entsteht ein "Beat"-Signal, dessen Frequenz die
Entfernung des Ziels verrät; die Phasenänderung von Chirp zu Chirp verrät die
Geschwindigkeit (Doppler-Effekt).

Wir erzeugen eine Matrix (n_chirps x n_samples) komplexer Abtastwerte – den
"Datenwürfel" einer Messung – plus realistisches Rauschen.
"""

from __future__ import annotations

import numpy as np

from .szenario import C, RadarConfig, Ziel


def erzeuge_signal(
    radar: RadarConfig,
    ziele: list[Ziel],
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Erzeugt den Datenwürfel (n_chirps x n_samples) für die gegebenen Ziele."""
    rng = rng or np.random.default_rng()
    n_s, n_c = radar.n_samples, radar.n_chirps

    fast = np.arange(n_s) / radar.fs          # Zeit innerhalb eines Chirps
    chirp_idx = np.arange(n_c)                 # Chirp-Nummer (slow time)

    cube = np.zeros((n_c, n_s), dtype=np.complex128)
    for z in ziele:
        # Beat-Frequenz aus der Entfernung
        f_beat = 2 * radar.slope * z.entfernung / C
        # Doppler-Frequenz aus der Geschwindigkeit
        f_dopp = 2 * z.geschwindigkeit * radar.f_c / C
        amplitude = np.sqrt(max(z.rcs, 1e-6))
        # Phase = Entfernungs-Term (über fast time) + Doppler-Term (über slow time)
        phase = 2 * np.pi * (
            f_beat * fast[None, :] + f_dopp * radar.t_pri * chirp_idx[:, None]
        )
        cube += amplitude * np.exp(1j * phase)

    # Komplexes Gauß-Rauschen passend zur gewünschten SNR hinzufügen
    signal_leistung = np.mean(np.abs(cube) ** 2) if ziele else 1.0
    rausch_leistung = signal_leistung / (10 ** (snr_db / 10))
    sigma = np.sqrt(rausch_leistung / 2)
    rauschen = sigma * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))
    return cube + rauschen
