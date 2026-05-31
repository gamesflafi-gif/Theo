"""Spielzug-Simulator: Offense-Play vs. Defense-Play.

Simuliert den Verlauf eines Spielzugs (Routen, Deckung, Pass-Rush), liefert
Trajektorien für die Animation und über viele Läufe eine Ausgangsverteilung.
"""

from theo.simulation.advisor import rank_defenses, rank_offenses
from theo.simulation.engine import Simulator, simulate_play
from theo.simulation.model import OutcomeDistribution, PlayerFrame, PlayResult
from theo.simulation.plays import (
    DEFENSE_LIBRARY,
    OFFENSE_LIBRARY,
    OFFENSE_SLOTS,
    ROUTE_NAMES,
    DefensePlay,
    OffensePlay,
    make_offense_play,
)

__all__ = [
    "Simulator", "simulate_play",
    "PlayResult", "OutcomeDistribution", "PlayerFrame",
    "OffensePlay", "DefensePlay",
    "OFFENSE_LIBRARY", "DEFENSE_LIBRARY", "OFFENSE_SLOTS", "ROUTE_NAMES",
    "make_offense_play", "rank_defenses", "rank_offenses",
]
