"""Kommandozeile für Theo Radar – rechnet ein komplettes Szenario durch.

Beispiel:
    python -m theo.radar.cli demo
    python -m theo.radar.cli demo --frames 30 --snr 15 --plots ausgabe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .signal import erzeuge_signal
from .szenario import RadarConfig, Ziel
from .tracking import MultiTracker
from .verarbeitung import finde_ziele, range_doppler_karte


def demo(frames: int = 25, snr_db: float = 18.0, dt: float = 0.1,
         plot_ordner: str | None = None, seed: int = 1) -> None:
    """Bewegte Ziele über mehrere Messungen erkennen und verfolgen."""
    radar = RadarConfig()
    rng = np.random.default_rng(seed)
    tracker = MultiTracker(dt=dt)

    # Drei gut getrennte Ziele mit Start-Entfernung und (konstanter) Geschwindigkeit
    ziele_start = [
        Ziel(15.0, 6.0, rcs=1.0, name="Drohne"),
        Ziel(85.0, -5.0, rcs=2.0, name="Fahrzeug"),
        Ziel(50.0, 0.0, rcs=0.6, name="stehendes Objekt"),
    ]

    print("Theo Radar – Demo")
    print(f"Auflösung: {radar.entfernung_aufloesung:.2f} m / "
          f"{radar.geschw_aufloesung:.2f} m/s | "
          f"Reichweite: {radar.entfernung_max:.0f} m, ±{radar.geschw_max:.0f} m/s")
    print(f"{frames} Messungen, SNR {snr_db:.0f} dB\n")

    letzte_leistung = letzte_r = letzte_v = None
    letzte_dets = []
    for f in range(frames):
        # Ziele bewegen sich (Entfernung ändert sich mit der Geschwindigkeit)
        aktuelle = []
        for z in ziele_start:
            neue_entf = z.entfernung + z.geschwindigkeit * dt * f
            if 1.0 < neue_entf < radar.entfernung_max - 1:
                aktuelle.append(Ziel(neue_entf, z.geschwindigkeit, z.rcs, z.name))

        cube = erzeuge_signal(radar, aktuelle, snr_db=snr_db, rng=rng)
        leistung, r_achse, v_achse = range_doppler_karte(cube, radar)
        dets = finde_ziele(leistung, r_achse, v_achse, pfa=1e-6)
        tracks = tracker.schritt(dets)
        letzte_leistung, letzte_r, letzte_v, letzte_dets = leistung, r_achse, v_achse, dets

        if f == frames - 1:
            print(f"Letzte Messung: {len(dets)} Detektionen, "
                  f"{len(tracks)} bestätigte Tracks:")
            for t in sorted(tracks, key=lambda t: t.entfernung):
                print(f"  Track {t.id}: {t.entfernung:5.1f} m | "
                      f"{t.geschwindigkeit:+5.1f} m/s | "
                      f"{t.treffer} Treffer")

    if plot_ordner:
        from .plot import zeichne_range_doppler, zeichne_tracks
        ordner = Path(plot_ordner)
        ordner.mkdir(parents=True, exist_ok=True)
        verlauf = {t.id: t.verlauf for t in tracker.tracks if t.bestaetigt}
        p1 = zeichne_range_doppler(letzte_leistung, letzte_r, letzte_v,
                                   letzte_dets, str(ordner / "range_doppler.png"))
        p2 = zeichne_tracks(verlauf, dt, str(ordner / "tracks.png"))
        print(f"\nBilder gespeichert: {p1} , {p2}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Theo Radar")
    sub = p.add_subparsers(dest="befehl", required=True)
    d = sub.add_parser("demo", help="Erkennung + Tracking über mehrere Messungen")
    d.add_argument("--frames", type=int, default=25)
    d.add_argument("--snr", type=float, default=18.0)
    d.add_argument("--dt", type=float, default=0.1)
    d.add_argument("--plots", default=None, help="Ordner für PNG-Ausgaben")
    d.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    if args.befehl == "demo":
        demo(args.frames, args.snr, args.dt, args.plots, args.seed)


if __name__ == "__main__":
    main()
