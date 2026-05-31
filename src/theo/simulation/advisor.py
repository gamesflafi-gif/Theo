"""Matchup-Berater: rankt Spielzüge über Monte-Carlo-Simulation.

Beantwortet "Welche Defense stoppt diesen Spielzug am besten?" bzw. "Welche
Offense schlägt diese Defense?" – indem ein Play gegen die ganze Bibliothek
simuliert und nach Ø-Raumgewinn sortiert wird.
"""

from __future__ import annotations

from theo.simulation.engine import Simulator
from theo.simulation.plays import (
    DEFENSE_LIBRARY,
    OFFENSE_LIBRARY,
    DefensePlay,
    OffensePlay,
)


def rank_defenses(off: OffensePlay, *, n: int = 80, base_seed: int = 0) -> list[dict]:
    """Alle Defenses gegen einen Offense-Play – beste (wenigste Yards) zuerst."""
    sim = Simulator()
    rows = []
    for did, deff in DEFENSE_LIBRARY.items():
        dist = sim.simulate_many(off, deff, n=n, base_seed=base_seed)
        rows.append({"id": did, "name": deff.name, **dist.to_dict()})
    rows.sort(key=lambda r: r["mean_yards"])  # weniger zugelassen = besser
    return rows


def rank_offenses(deff: DefensePlay, *, n: int = 80, base_seed: int = 0) -> list[dict]:
    """Alle Offense-Plays gegen eine Defense – bester (meiste Yards) zuerst."""
    sim = Simulator()
    rows = []
    for oid, off in OFFENSE_LIBRARY.items():
        dist = sim.simulate_many(off, deff, n=n, base_seed=base_seed)
        rows.append({"id": oid, "name": off.name, **dist.to_dict()})
    rows.sort(key=lambda r: -r["mean_yards"])  # mehr Raumgewinn = besser
    return rows


__all__ = ["rank_defenses", "rank_offenses"]
