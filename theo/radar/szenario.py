"""Radar-Parameter und Ziel-Definitionen."""

from __future__ import annotations

from dataclasses import dataclass, field

C = 3.0e8  # Lichtgeschwindigkeit in m/s


@dataclass
class RadarConfig:
    """Parameter eines FMCW-Radars (Standardwerte ~ 77-GHz-Sensorik)."""

    f_c: float = 77e9          # Trägerfrequenz [Hz]
    bandbreite: float = 200e6  # Sweep-Bandbreite B [Hz]  -> Entfernungsauflösung
    t_chirp: float = 40e-6     # Dauer eines Chirps [s]
    t_pri: float = 50e-6       # Chirp-Wiederholzeit (PRI) [s] -> Geschw.-Auflösung
    n_samples: int = 256       # Abtastwerte pro Chirp ("fast time")
    n_chirps: int = 128        # Chirps pro Messung ("slow time")

    @property
    def slope(self) -> float:
        """Frequenzanstieg des Chirps [Hz/s]."""
        return self.bandbreite / self.t_chirp

    @property
    def fs(self) -> float:
        """Abtastrate [Hz]."""
        return self.n_samples / self.t_chirp

    @property
    def wellenlaenge(self) -> float:
        return C / self.f_c

    @property
    def entfernung_aufloesung(self) -> float:
        """Kleinste unterscheidbare Entfernung [m] = c / (2B)."""
        return C / (2 * self.bandbreite)

    @property
    def entfernung_max(self) -> float:
        """Nutzbare Maximalentfernung [m].

        Wir behalten nach der Range-FFT nur die positive Spektrumshälfte
        (n_samples/2 Bins), daher ist die nutzbare Reichweite halb so groß wie
        die theoretische Nyquist-Grenze.
        """
        return (self.n_samples // 2) * self.entfernung_aufloesung

    @property
    def geschw_aufloesung(self) -> float:
        """Kleinste unterscheidbare Geschwindigkeit [m/s]."""
        return self.wellenlaenge / (2 * self.n_chirps * self.t_pri)

    @property
    def geschw_max(self) -> float:
        """Maximal eindeutige (radiale) Geschwindigkeit [m/s], ±."""
        return self.wellenlaenge / (4 * self.t_pri)


@dataclass
class Ziel:
    """Ein Radarziel."""

    entfernung: float          # [m]
    geschwindigkeit: float     # radiale Geschwindigkeit [m/s], + = entfernt sich
    rcs: float = 1.0           # Radarrückstreuquerschnitt (relative Stärke)
    name: str = ""


@dataclass
class Szenario:
    """Ein Messszenario: ein Radar und mehrere Ziele."""

    radar: RadarConfig = field(default_factory=RadarConfig)
    ziele: list[Ziel] = field(default_factory=list)
