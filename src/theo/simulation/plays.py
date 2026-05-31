"""Routen, Formationen und Spielzug-Bibliothek für den Simulator.

Eine OffensePlay ordnet jedem Skill-Spieler eine Route (oder Lauf/Block) zu, eine
DefensePlay legt das Deckungsschema fest. Beides ist datengetrieben, sodass sich
über die Oberfläche eigene Spielzüge zusammenstellen lassen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from theo.simulation.model import FIELD_CENTER

# --- Routen (Route Tree) ---------------------------------------------------
# Wegpunkte als (dx, dy) in Yards, relativ zur Startposition eines Receivers
# auf der RECHTEN Seite (dx>0 = Richtung Seitenlinie/außen, dx<0 = innen).
# Für links aufgestellte Receiver wird dx gespiegelt.
ROUTE_WAYPOINTS: dict[str, list[tuple[float, float]]] = {
    "go": [(0, 22)],
    "hitch": [(0, 6), (0, 5)],
    "slant": [(-0.5, 1.5), (-5, 6)],
    "out": [(0, 6), (5.5, 6.5)],
    "in": [(0, 8), (-6, 8.5)],
    "curl": [(0, 11), (-1.5, 10)],
    "post": [(0, 8), (-7, 17)],
    "corner": [(0, 8), (6, 16)],
    "flat": [(2.5, 1), (8, 1.5)],
    "screen": [(-2, -1), (-7, -1)],
    "comeback": [(0, 13), (1.5, 11)],
    "block": [],
    "run": [],
}

ROUTE_NAMES = [r for r in ROUTE_WAYPOINTS if r not in ("block", "run")]


# --- Formation -------------------------------------------------------------
# Standard-Startaufstellung (Shotgun, 11 Personnel-artig). y=0 ist die Line of
# Scrimmage; Offense läuft Richtung +y.
@dataclass(frozen=True)
class Slot:
    id: str
    role: str
    x: float
    y: float
    side: int  # +1 rechts der Mitte, -1 links (für Routen-Spiegelung)


def _slot(id, role, x, y):
    return Slot(id, role, x, y, side=1 if x >= FIELD_CENTER else -1)


OFFENSE_SLOTS: list[Slot] = [
    _slot("QB", "QB", FIELD_CENTER, -5.0),
    _slot("RB", "RB", FIELD_CENTER - 1.5, -6.0),
    _slot("WR_L", "WR", 6.0, -1.0),
    _slot("WR_R", "WR", 47.0, -1.0),
    _slot("SLOT", "WR", 37.0, -2.0),
    _slot("TE", "TE", FIELD_CENTER + 4.5, 0.0),
]
# Offensive Line (Blocker, v. a. statisch – für Pass-Schutz/Optik).
OL_SLOTS: list[Slot] = [
    _slot(f"OL{i}", "OL", FIELD_CENTER + dx, 0.0)
    for i, dx in enumerate((-3.0, -1.5, 0.0, 1.5, 3.0))
]

# Verteidiger-Startaufstellung: 4 DL, 3 LB, 2 CB, 2 S = 11.
DEFENSE_SLOTS: list[Slot] = [
    _slot("DL1", "DL", FIELD_CENTER - 4.5, 1.0),
    _slot("DL2", "DL", FIELD_CENTER - 1.5, 1.0),
    _slot("DL3", "DL", FIELD_CENTER + 1.5, 1.0),
    _slot("DL4", "DL", FIELD_CENTER + 4.5, 1.0),
    _slot("LB1", "LB", FIELD_CENTER - 5.0, 4.5),
    _slot("LB2", "LB", FIELD_CENTER, 5.0),
    _slot("LB3", "LB", FIELD_CENTER + 5.0, 4.5),
    _slot("CB_L", "CB", 6.0, 6.0),
    _slot("CB_R", "CB", 47.0, 6.0),
    _slot("S_L", "S", FIELD_CENTER - 8.0, 12.0),
    _slot("S_R", "S", FIELD_CENTER + 8.0, 12.0),
]

OFFENSE_SLOT_IDS = [s.id for s in OFFENSE_SLOTS]


# --- Plays -----------------------------------------------------------------
@dataclass
class OffensePlay:
    id: str
    name: str
    kind: str                       # "pass" | "run"
    routes: dict[str, str]          # Slot-ID -> Routenname (oder "block"/"run")
    description: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "routes": self.routes, "description": self.description}


@dataclass
class DefensePlay:
    id: str
    name: str
    coverage: str                   # "man1" | "cover2" | "cover3"
    blitz: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "coverage": self.coverage,
                "blitz": self.blitz, "description": self.description}


# Vordefinierte Offense-Spielzüge.
OFFENSE_LIBRARY: dict[str, OffensePlay] = {
    p.id: p for p in [
        OffensePlay("slant_flat", "Slant-Flat", "pass",
                    {"WR_L": "slant", "WR_R": "slant", "SLOT": "flat",
                     "TE": "hitch", "RB": "flat"},
                    "Schnelle Slants mit Flat darunter – gut gegen Blitz."),
        OffensePlay("four_verticals", "Four Verticals", "pass",
                    {"WR_L": "go", "WR_R": "go", "SLOT": "go", "TE": "post",
                     "RB": "block"},
                    "Vier tiefe Routen – dehnt die Coverage vertikal."),
        OffensePlay("mesh", "Mesh", "pass",
                    {"WR_L": "in", "WR_R": "in", "SLOT": "corner",
                     "TE": "flat", "RB": "flat"},
                    "Kreuzende Unterrouten – stark gegen Mann-Deckung."),
        OffensePlay("smash", "Smash", "pass",
                    {"WR_L": "hitch", "WR_R": "hitch", "SLOT": "corner",
                     "TE": "corner", "RB": "block"},
                    "Hitch-Corner-Kombi – High-Low gegen die Corners."),
        OffensePlay("pa_post", "Play-Action Post", "pass",
                    {"WR_L": "post", "WR_R": "comeback", "SLOT": "out",
                     "TE": "block", "RB": "block"},
                    "Tiefer Post nach Play-Action – braucht Zeit."),
        OffensePlay("inside_zone", "Inside Zone (Lauf)", "run",
                    {"WR_L": "block", "WR_R": "block", "SLOT": "block",
                     "TE": "block", "RB": "run"},
                    "Laufspielzug zwischen den Tackles."),
    ]
}

# Vordefinierte Defense-Spielzüge.
DEFENSE_LIBRARY: dict[str, DefensePlay] = {
    p.id: p for p in [
        DefensePlay("cover1_man", "Cover 1 (Mann)", "man1", False,
                    "Mann-Deckung mit einem freien Safety in der Tiefe."),
        DefensePlay("cover2_zone", "Cover 2 (Zone)", "cover2", False,
                    "Zwei tiefe Safeties, Corners in den Flats."),
        DefensePlay("cover3_zone", "Cover 3 (Zone)", "cover3", False,
                    "Drei tiefe Zonen, vier Verteidiger darunter."),
        DefensePlay("man_blitz", "Cover 1 Blitz", "man1", True,
                    "Mann-Deckung mit zusätzlichem Blitzer – früher Druck."),
        DefensePlay("cover2_blitz", "Cover 2 Blitz", "cover2", True,
                    "Zone-Blitz – Druck bei zwei tiefen Zonen."),
    ]
}


def route_waypoints(route: str, start_x: float, start_y: float, side: int):
    """Absolute Wegpunkte einer Route ab der Startposition (mit Seiten-Spiegelung)."""
    pts = ROUTE_WAYPOINTS.get(route, [])
    out = []
    for dx, dy in pts:
        out.append((start_x + dx * side, start_y + dy))
    return out


def make_offense_play(routes: dict[str, str], *, kind: str = "pass",
                      name: str = "Eigener Spielzug") -> OffensePlay:
    """Baut aus einer Routen-Zuordnung einen individuellen Offense-Spielzug."""
    clean = {sid: routes.get(sid, "block") for s in OFFENSE_SLOTS for sid in [s.id]}
    if kind == "run":
        clean["RB"] = "run"
    return OffensePlay("custom", name, kind, clean, "Individuell zusammengestellt.")


__all__ = [
    "ROUTE_WAYPOINTS", "ROUTE_NAMES", "Slot",
    "OFFENSE_SLOTS", "OL_SLOTS", "DEFENSE_SLOTS", "OFFENSE_SLOT_IDS",
    "OffensePlay", "DefensePlay", "OFFENSE_LIBRARY", "DEFENSE_LIBRARY",
    "route_waypoints", "make_offense_play",
]
