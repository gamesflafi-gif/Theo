"""Simulations-Engine: ein Offense-Play gegen ein Defense-Play.

Tick-basiertes, probabilistisches Modell. Bewegt Spieler über Routen, lässt die
Verteidigung in Mann-/Zone-Deckung reagieren, modelliert Pass-Rush-Druck und löst
den Pass über die Separation am Catch-Punkt auf. Liefert Trajektorien für die
Animation und – über viele Läufe – eine Ausgangsverteilung.

Bewusst vereinfacht: keine echte Physik, sondern ein plausibles Spielmodell.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from theo.simulation.model import (
    FIELD_CENTER,
    OutcomeDistribution,
    PlayerFrame,
    PlayResult,
)
from theo.simulation.plays import (
    DEFENSE_SLOTS,
    OFFENSE_SLOTS,
    OL_SLOTS,
    DefensePlay,
    OffensePlay,
    route_waypoints,
)

DT = 0.1
BALL_SPEED = 19.0
CATCH_RADIUS = 1.2
TACKLE_RADIUS = 1.1

SPEED = {"QB": 7.2, "RB": 8.0, "WR": 8.7, "TE": 7.8, "OL": 5.0,
         "DL": 7.0, "LB": 7.6, "CB": 8.5, "S": 8.1}


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _move_toward(x, y, tx, ty, max_step):
    d = _dist(x, y, tx, ty)
    if d <= max_step or d == 0:
        return tx, ty, True
    f = max_step / d
    return x + (tx - x) * f, y + (ty - y) * f, False


class _P:
    """Veränderlicher Spielerzustand während der Simulation."""

    __slots__ = ("id", "role", "team", "x", "y", "speed", "waypoints", "wp",
                 "behavior", "target", "landmark", "reaction", "vx", "vy")

    def __init__(self, id, role, team, x, y, speed):
        self.id, self.role, self.team = id, role, team
        self.x, self.y, self.speed = x, y, speed
        self.waypoints: list[tuple[float, float]] = []
        self.wp = 0
        self.behavior = ""
        self.target = ""
        self.landmark = (x, y)
        self.reaction = 0.0
        self.vx, self.vy = 0.0, 0.0

    def frame(self) -> PlayerFrame:
        return PlayerFrame(self.id, self.x, self.y, self.team, self.role)


class Simulator:
    def __init__(self, *, dt: float = DT, max_time: float = 6.0):
        self.dt = dt
        self.max_time = max_time

    # -- Aufbau --------------------------------------------------------------
    def _build(self, off: OffensePlay, rng: random.Random):
        players: dict[str, _P] = {}
        for s in OFFENSE_SLOTS:
            p = _P(s.id, s.role, "offense", s.x, s.y, SPEED[s.role])
            route = off.routes.get(s.id, "block")
            if route not in ("block", "run") and s.id != "QB":
                p.waypoints = route_waypoints(route, s.x, s.y, s.side)
                p.behavior = "route"
            elif route == "run":
                p.behavior = "run"
            else:
                p.behavior = "block"
            if s.id == "QB":
                p.behavior = "qb"
            # leichte individuelle Geschwindigkeitsstreuung
            p.speed *= rng.uniform(0.95, 1.05)
            players[s.id] = p
        for s in OL_SLOTS:
            players[s.id] = _P(s.id, s.role, "offense", s.x, s.y, SPEED["OL"])
        for s in DEFENSE_SLOTS:
            p = _P(s.id, s.role, "defense", s.x, s.y, SPEED[s.role])
            p.reaction = rng.uniform(0.15, 0.45)
            p.speed *= rng.uniform(0.95, 1.05)
            players[s.id] = p
        return players

    def _assign_defense(self, players, off: OffensePlay, deff: DefensePlay,
                        rng: random.Random):
        """Legt Rush/Mann/Zone je Verteidiger nach Coverage fest."""
        rushers = ["DL1", "DL2", "DL3", "DL4"]
        cov = deff.coverage
        # Receiver, die tatsächlich eine Route laufen.
        eligible = [s.id for s in OFFENSE_SLOTS
                    if s.id != "QB" and off.routes.get(s.id) not in ("block", "run")]

        if cov == "man1":
            man = {"CB_L": "WR_L", "CB_R": "WR_R", "S_R": "SLOT",
                   "LB2": "TE", "LB3": "RB"}
            for did, tid in man.items():
                players[did].behavior = "man"
                players[did].target = tid
            players["S_L"].behavior = "zone"     # freier Safety, tiefe Mitte
            players["S_L"].landmark = (FIELD_CENTER, 17.0)
            players["LB1"].behavior = "spy"
            players["LB1"].landmark = (FIELD_CENTER, 5.0)
            if deff.blitz:
                players["LB1"].behavior = "rush"
                rushers.append("LB1")
        elif cov == "cover2":
            zones = {"S_L": (14.0, 18.0), "S_R": (39.0, 18.0),
                     "CB_L": (8.0, 5.0), "CB_R": (45.0, 5.0),
                     "LB1": (18.0, 9.0), "LB2": (FIELD_CENTER, 10.0),
                     "LB3": (35.0, 9.0)}
            for did, lm in zones.items():
                players[did].behavior = "zone"
                players[did].landmark = lm
            if deff.blitz:
                players["LB2"].behavior = "rush"
                rushers.append("LB2")
        elif cov == "cover0":
            # Mann ohne tiefe Hilfe, alle übrigen blitzen.
            man = {"CB_L": "WR_L", "CB_R": "WR_R", "S_R": "SLOT",
                   "LB2": "TE", "LB3": "RB"}
            for did, tid in man.items():
                players[did].behavior = "man"
                players[did].target = tid
            players["S_L"].behavior = "rush"
            players["LB1"].behavior = "rush"
            rushers += ["S_L", "LB1"]
        elif cov == "cover4":
            zones = {"CB_L": (9.0, 17.0), "S_L": (19.0, 18.0),
                     "S_R": (35.0, 18.0), "CB_R": (44.0, 17.0),
                     "LB1": (15.0, 8.0), "LB2": (FIELD_CENTER, 9.0),
                     "LB3": (39.0, 8.0)}
            for did, lm in zones.items():
                players[did].behavior = "zone"
                players[did].landmark = lm
            if deff.blitz:
                players["LB2"].behavior = "rush"
                rushers.append("LB2")
        else:  # cover3
            zones = {"CB_L": (9.0, 18.0), "CB_R": (44.0, 18.0),
                     "S_L": (FIELD_CENTER, 19.0), "S_R": (38.0, 8.0),
                     "LB1": (15.0, 9.0), "LB2": (FIELD_CENTER, 9.0),
                     "LB3": (40.0, 9.0)}
            for did, lm in zones.items():
                players[did].behavior = "zone"
                players[did].landmark = lm
            if deff.blitz:
                players["LB1"].behavior = "rush"
                rushers.append("LB1")

        for rid in rushers:
            players[rid].behavior = "rush"
        return rushers, eligible

    # -- Bewegung pro Tick ---------------------------------------------------
    def _step_offense(self, players, t):
        for s in OFFENSE_SLOTS:
            p = players[s.id]
            step = p.speed * self.dt
            if p.behavior == "qb":
                # Dropback in den ersten ~0.8s, dann im Pocket halten.
                tx, ty = FIELD_CENTER, -7.0
                p.x, p.y, _ = _move_toward(p.x, p.y, tx, ty, step * 0.7)
            elif p.behavior in ("route", "run"):
                self._advance_route(p, step)
            elif p.behavior == "block":
                pass  # Blocker bleiben (vereinfachte Pass-Protection)

    def _advance_route(self, p: _P, step):
        if p.wp < len(p.waypoints):
            tx, ty = p.waypoints[p.wp]
            p.vx, p.vy = tx - p.x, ty - p.y
            p.x, p.y, reached = _move_toward(p.x, p.y, tx, ty, step)
            if reached:
                p.wp += 1
        else:
            # Nach der letzten Marke: vorwärts laufende Routen driften langsam
            # weiter; zurückkommende Routen (Hitch/Curl/Comeback) bleiben im
            # Fenster stehen.
            if p.vy >= 0:
                n = math.hypot(p.vx, p.vy) or 1.0
                p.x += (p.vx / n) * step
                p.y += (p.vy / n) * step

    def _step_defense(self, players, t, ball_thrown, ball_pos):
        for s in DEFENSE_SLOTS:
            p = players[s.id]
            step = p.speed * self.dt
            b = p.behavior
            if b == "rush":
                p.x, p.y, _ = _move_toward(p.x, p.y, FIELD_CENTER, -7.0, step * 0.6)
            elif b == "man":
                tgt = players[p.target]
                if t < p.reaction:
                    p.y += step * 0.4  # kurzer Backpedal / Reaktionszeit
                else:
                    p.x, p.y, _ = _move_toward(p.x, p.y, tgt.x, tgt.y, step)
            elif b in ("zone", "spy"):
                lx, ly = p.landmark
                if ball_thrown and ball_pos is not None and \
                        _dist(p.x, p.y, ball_pos[0], ball_pos[1]) < 13.0:
                    # Auf den Ball/Catch-Punkt brechen, wenn nah genug.
                    lx, ly = ball_pos
                elif ly >= 14.0:
                    # Tiefe Zone: einen vertikal laufenden Receiver mitnehmen.
                    deep = [players[o.id] for o in OFFENSE_SLOTS
                            if o.id != "QB" and abs(players[o.id].x - lx) < 9.0
                            and players[o.id].y > 6.0]
                    if deep:
                        tgt = min(deep, key=lambda r: abs(r.x - lx))
                        lx = (tgt.x + lx) / 2.0
                        ly = max(ly, tgt.y - 1.5)   # Leverage oben halten
                p.x, p.y, _ = _move_toward(p.x, p.y, lx, ly, step)

    # -- Hauptsimulation -----------------------------------------------------
    def simulate(self, off: OffensePlay, deff: DefensePlay, *, seed: int = 0,
                 record_frames: bool = True) -> PlayResult:
        rng = random.Random(seed)
        players = self._build(off, rng)
        rushers, eligible = self._assign_defense(players, off, deff, rng)

        if off.kind == "run":
            return self._simulate_run(players, off, deff, rng, record_frames)

        num_blockers = len(OL_SLOTS) + sum(
            1 for s in OFFENSE_SLOTS
            if s.id in ("RB", "TE") and off.routes.get(s.id) == "block"
        )
        pressure_time = 2.6 + 0.30 * (num_blockers - len(rushers)) + rng.uniform(-0.3, 0.3)
        pressure_time = max(1.3, min(3.7, pressure_time))

        frames: list[list[PlayerFrame]] = []
        ball: list = []
        earliest, threshold = 1.0, 1.8
        t = 0.0
        thrown = False
        target_id = None

        def record(bp):
            if record_frames:
                frames.append([players[s.id].frame() for s in
                               OFFENSE_SLOTS + OL_SLOTS + DEFENSE_SLOTS])
                ball.append(bp)

        # Phase 1: vor dem Wurf.
        while t <= pressure_time + 1e-9:
            self._step_offense(players, t)
            self._step_defense(players, t, False, None)
            record(None)
            if t >= earliest and eligible:
                tid = self._best_target(players, eligible, t)
                if tid is not None:
                    sep = self._separation(players, tid)
                    developed = players[tid].y > 1.0 or t > 1.4
                    if sep >= threshold and developed:
                        thrown, target_id = True, tid
                        break
            t += self.dt

        if not thrown:
            tid = self._best_target(players, eligible, t) if eligible else None
            if tid is None or self._separation(players, tid) < 1.0:
                if tid is None or rng.random() < 0.6:
                    return self._finish_sack(frames, ball, players, rng, record_frames)
            target_id = tid

        throw_time = t  # Zeitpunkt des Wurfs festhalten (vor dem Ballflug).
        return self._resolve_pass(players, target_id, throw_time, t, frames, ball,
                                  rng, record_frames, record)

    def _best_target(self, players, eligible, t):
        best, best_score = None, -1e9
        for tid in eligible:
            p = players[tid]
            sep = self._separation(players, tid)
            # Tiefe belohnen (gedeckelt bei 12 Yd), Würfe hinter die LOS bestrafen.
            depth_term = 0.13 * max(-6.0, min(p.y, 12.0))
            score = sep + depth_term
            if score > best_score:
                best, best_score = tid, score
        return best

    def _separation(self, players, tid) -> float:
        p = players[tid]
        return min(_dist(p.x, p.y, players[s.id].x, players[s.id].y)
                   for s in DEFENSE_SLOTS)

    def _nearest_defender(self, players, x, y):
        return min((players[s.id] for s in DEFENSE_SLOTS),
                   key=lambda d: _dist(d.x, d.y, x, y))

    def _resolve_pass(self, players, target_id, throw_time, t, frames, ball, rng,
                      record_frames, record):
        receiver = players[target_id]
        qb = players["QB"]
        ball_x, ball_y = qb.x, qb.y
        air_ticks = 0
        max_air = int(2.6 / self.dt)
        caught = False
        while air_ticks < max_air:
            self._step_offense(players, t)
            # Verteidiger brechen auf den Catch-Punkt (Receiver), nicht auf die
            # aktuelle Ballposition (die anfangs noch hinten beim QB liegt).
            self._step_defense(players, t, True, (receiver.x, receiver.y))
            # Ball Richtung aktueller Receiver-Position (Homing -> Vorhalten).
            ball_x, ball_y, reached = _move_toward(
                ball_x, ball_y, receiver.x, receiver.y, BALL_SPEED * self.dt)
            record((ball_x, ball_y))
            if reached or _dist(ball_x, ball_y, receiver.x, receiver.y) <= CATCH_RADIUS:
                caught = True
                break
            t += self.dt
            air_ticks += 1

        sep = self._separation(players, target_id)
        # Ausführungsfehler (Überwurf/Drop) -> nie 100% sicher, nie chancenlos.
        p_complete = max(0.03, min(0.93, _logistic((sep - 1.2) * 1.4)))
        roll = rng.random()
        tt = round(throw_time, 1)

        if caught and roll < p_complete:
            yards = self._run_after_catch(players, target_id, t, frames, ball,
                                          rng, record_frames, record)
            res = PlayResult("complete", yards, self.dt, target=target_id,
                             throw_time=tt, frames=frames, ball=ball)
            res.notes.append(f"Separation am Catch: {sep:.1f} Yards "
                             f"(p={p_complete:.0%}).")
            return res

        # Inkomplett – ggf. Interception bei sehr enger Deckung.
        if sep < 0.7 and rng.random() < 0.18:
            return PlayResult("interception", 0.0, self.dt, target=target_id,
                              throw_time=tt, frames=frames, ball=ball,
                              notes=[f"Enge Deckung ({sep:.1f} Yd) – abgefangen."])
        return PlayResult("incomplete", 0.0, self.dt, target=target_id,
                          throw_time=tt, frames=frames, ball=ball,
                          notes=[f"Separation {sep:.1f} Yd, p={p_complete:.0%}."])

    def _run_after_catch(self, players, target_id, t, frames, ball, rng,
                         record_frames, record) -> float:
        receiver = players[target_id]
        for _ in range(int(1.6 / self.dt)):
            receiver.y += receiver.speed * 0.8 * self.dt
            self._step_defense(players, t, True, (receiver.x, receiver.y))
            record((receiver.x, receiver.y))
            nd = self._nearest_defender(players, receiver.x, receiver.y)
            if _dist(nd.x, nd.y, receiver.x, receiver.y) <= TACKLE_RADIUS:
                break
            t += self.dt
        return receiver.y

    def _finish_sack(self, frames, ball, players, rng, record_frames):
        yards = -rng.uniform(4.0, 8.0)
        return PlayResult("sack", yards, self.dt, frames=frames, ball=ball,
                          notes=["Kein Receiver rechtzeitig frei – Druck führt zum Sack."])

    def _simulate_run(self, players, off, deff, rng, record_frames):
        box = sum(1 for s in DEFENSE_SLOTS
                  if players[s.id].behavior in ("rush", "spy")
                  or players[s.id].y < 6.0)
        # Nur Linemen und ein blockender Tight End zählen als Run-Blocker.
        blockers = len(OL_SLOTS) + (1 if off.routes.get("TE") == "block" else 0)
        base = 4.0 + 0.7 * (blockers - box) + rng.gauss(0, 2.2)
        if deff.blitz:
            base += rng.uniform(-1.0, 3.0)  # Blitz: riskanter (Lücke o. Stop)
        if rng.random() < 0.10:
            base += rng.uniform(8.0, 24.0)  # Ausbruch
        yards = max(-4.0, base)

        frames: list[list[PlayerFrame]] = []
        ball: list = []
        rb = players["RB"]
        gap_x = FIELD_CENTER + rng.uniform(-3.5, 3.5)
        path = [(gap_x, 0.0), (gap_x, yards)]
        wp = 0
        t = 0.0
        while wp < len(path) and t < self.max_time:
            self._step_defense(players, t, True, (rb.x, rb.y))
            tx, ty = path[wp]
            rb.x, rb.y, reached = _move_toward(rb.x, rb.y, tx, ty, rb.speed * self.dt)
            if reached:
                wp += 1
            if record_frames:
                frames.append([players[s.id].frame() for s in
                               OFFENSE_SLOTS + OL_SLOTS + DEFENSE_SLOTS])
                ball.append((rb.x, rb.y))
            t += self.dt
        return PlayResult("run", round(yards, 1), self.dt, target="RB",
                          frames=frames, ball=ball,
                          notes=[f"Box: {box} Verteidiger, Blocker: {blockers}."])

    # -- Monte Carlo ---------------------------------------------------------
    def simulate_many(self, off: OffensePlay, deff: DefensePlay, *,
                      n: int = 100, base_seed: int = 0) -> OutcomeDistribution:
        outcomes: Counter = Counter()
        yards: list[float] = []
        for i in range(n):
            res = self.simulate(off, deff, seed=base_seed + i, record_frames=False)
            outcomes[res.outcome] += 1
            yards.append(res.yards)
        return OutcomeDistribution(n=n, outcomes=outcomes, yards=yards)


def simulate_play(off: OffensePlay, deff: DefensePlay, *, seed: int = 0) -> PlayResult:
    return Simulator().simulate(off, deff, seed=seed)


__all__ = ["Simulator", "simulate_play", "DT"]
