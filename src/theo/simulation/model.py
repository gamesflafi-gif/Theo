"""Datenmodell des Spielzug-Simulators.

Vereinfachtes, probabilistisches Modell – keine echte Physik. Es bildet Routen,
Deckung und Pass-Rush plausibel ab, um mögliche Ausgänge eines Spielzugs zu
schätzen und ihn animieren zu können.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

# Feldmaße (Yards).
FIELD_WIDTH = 53.3
FIELD_CENTER = FIELD_WIDTH / 2.0


@dataclass
class PlayerFrame:
    """Position eines Spielers in einem Zeitschritt (für die Animation)."""

    id: str
    x: float
    y: float
    team: str        # "offense" | "defense"
    role: str        # z. B. "QB", "WR", "CB", "DL" – für Farbe/Label

    def as_dict(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2),
                "team": self.team, "role": self.role}


@dataclass
class PlayResult:
    """Ergebnis einer einzelnen Simulation eines Spielzugs."""

    outcome: str             # complete | incomplete | sack | interception | run
    yards: float
    dt: float
    target: str | None = None
    throw_time: float | None = None
    frames: list[list[PlayerFrame]] = field(default_factory=list)
    ball: list[tuple[float, float] | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        label = {
            "complete": "Pass komplett",
            "incomplete": "Pass inkomplett",
            "sack": "Sack",
            "interception": "Interception",
            "run": "Lauf",
        }.get(self.outcome, self.outcome)
        parts = [f"{label}: {self.yards:+.1f} Yards"]
        if self.target:
            parts.append(f"Ziel: {self.target}")
        if self.throw_time is not None:
            parts.append(f"Wurf bei {self.throw_time:.1f}s")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "yards": round(self.yards, 1),
            "target": self.target,
            "throw_time": self.throw_time,
            "dt": self.dt,
            "summary": self.summary(),
            "notes": self.notes,
            "frames": [[p.as_dict() for p in step] for step in self.frames],
            "ball": [
                ({"x": round(b[0], 2), "y": round(b[1], 2)} if b else None)
                for b in self.ball
            ],
        }


@dataclass
class OutcomeDistribution:
    """Aggregierte Statistik über viele Simulationen (Monte Carlo)."""

    n: int
    outcomes: Counter
    yards: list[float]

    @property
    def mean_yards(self) -> float:
        return statistics.mean(self.yards) if self.yards else 0.0

    @property
    def median_yards(self) -> float:
        return statistics.median(self.yards) if self.yards else 0.0

    @property
    def stdev_yards(self) -> float:
        return statistics.pstdev(self.yards) if len(self.yards) > 1 else 0.0

    def pct(self, outcome: str) -> float:
        return 100.0 * self.outcomes.get(outcome, 0) / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_yards": round(self.mean_yards, 2),
            "median_yards": round(self.median_yards, 2),
            "stdev_yards": round(self.stdev_yards, 2),
            "outcomes": dict(self.outcomes),
            "outcome_pct": {k: round(self.pct(k), 1) for k in self.outcomes},
            "best_yards": round(max(self.yards), 1) if self.yards else 0.0,
            "worst_yards": round(min(self.yards), 1) if self.yards else 0.0,
        }

    def summary(self) -> str:
        lines = [
            f"{self.n} Simulationen – Ø {self.mean_yards:.1f} Yards "
            f"(Median {self.median_yards:.1f}, σ {self.stdev_yards:.1f})",
        ]
        for outcome, cnt in self.outcomes.most_common():
            lines.append(f"  {outcome}: {cnt} ({self.pct(outcome):.0f}%)")
        return "\n".join(lines)


__all__ = [
    "FIELD_WIDTH",
    "FIELD_CENTER",
    "PlayerFrame",
    "PlayResult",
    "OutcomeDistribution",
]
