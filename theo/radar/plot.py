"""Visualisierung der Radar-Ergebnisse (speichert PNG-Bilder)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # ohne Bildschirm, direkt in Dateien rendern
import matplotlib.pyplot as plt
import numpy as np


def zeichne_range_doppler(leistung, entfernung_achse, geschw_achse,
                          detektionen, pfad: str) -> str:
    """Speichert die Range-Doppler-Karte mit eingezeichneten Detektionen."""
    fig, ax = plt.subplots(figsize=(8, 6))
    db = 10 * np.log10(leistung / leistung.max() + 1e-12)
    bild = ax.imshow(
        db, aspect="auto", origin="lower", cmap="viridis",
        extent=[geschw_achse[0], geschw_achse[-1], entfernung_achse[0], entfernung_achse[-1]],
        vmin=-40, vmax=0,
    )
    for d in detektionen:
        ax.plot(d.geschwindigkeit, d.entfernung, "rx", markersize=12, markeredgewidth=2)
        ax.annotate(f"{d.entfernung:.0f} m\n{d.geschwindigkeit:+.1f} m/s",
                    (d.geschwindigkeit, d.entfernung), color="white",
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Geschwindigkeit [m/s]")
    ax.set_ylabel("Entfernung [m]")
    ax.set_title("Theo Radar – Range-Doppler-Karte (rote X = erkannte Ziele)")
    fig.colorbar(bild, ax=ax, label="Leistung [dB]")
    fig.tight_layout()
    fig.savefig(pfad, dpi=110)
    plt.close(fig)
    return pfad


def zeichne_tracks(track_verlauf: dict, dt: float, pfad: str) -> str:
    """Speichert den Entfernungsverlauf aller Tracks über die Zeit."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for track_id, punkte in track_verlauf.items():
        if len(punkte) < 2:
            continue
        zeiten = np.arange(len(punkte)) * dt
        entfernungen = [p[0] for p in punkte]
        ax.plot(zeiten, entfernungen, "-o", markersize=3, label=f"Track {track_id}")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Entfernung [m]")
    ax.set_title("Theo Radar – verfolgte Ziele über die Zeit")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pfad, dpi=110)
    plt.close(fig)
    return pfad
